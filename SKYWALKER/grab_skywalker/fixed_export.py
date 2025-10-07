#!/usr/bin/env python3
import argparse, os, re, sys, pickle
import torch

EPS = 1e-8

# ----------------------------- utils ---------------------------------
def _to_f32(t):
    return t.float() if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.float32)

def _dump_stats(out_path, rm, rv):
    sidecar = os.path.splitext(out_path)[0] + "_obs_stats.pt"
    torch.save({"running_mean": rm.cpu(), "running_var": rv.cpu()}, sidecar)
    print(f"💾 Wrote sidecar stats: {sidecar}")

def _list_keys(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            kp = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                yield from _list_keys(v, kp)
            else:
                yield kp, v

def _try_load(path):
    """
    Try torch.load first (with weights_only=False due to PyTorch 2.6 change).
    If that fails, try pickle.load.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e1:
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e2:
            raise RuntimeError(f"Failed to load '{path}' via torch.load or pickle: {e1} | {e2}")

# -------------------- find policy state_dict in ckpt ------------------
def _extract_policy_state(raw):
    """
    Try common SKRL layouts and fallbacks. Returns a flat state_dict for the policy.
    """
    # 1) Common direct keys
    direct_candidates = [
        "policy", "actor", "policy_model", "models/policy",
        "agents/0/policy", "agent/policy",
        "actor_network", "pi", "pi/model",
    ]
    for k in direct_candidates:
        node = raw
        for part in k.split("/"):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                node = None
                break
        if isinstance(node, dict) and any(isinstance(v, torch.Tensor) for v in node.values()):
            return node

    # 2) If the top-level *is* a state_dict (flat)
    if isinstance(raw, dict) and any(isinstance(v, torch.Tensor) for v in raw.values()):
        # Heuristic: contains typical policy keys
        if any("policy_layer" in s or "actor" in s or "net_container" in s for s in raw.keys()):
            return raw

    # 3) Walk and find something that looks like the policy block
    best = None
    for k, v in _list_keys(raw):
        if isinstance(v, dict) and any(isinstance(t, torch.Tensor) for t in v.values()):
            ks = list(v.keys())
            if any("policy_layer" in s for s in ks) or any(re.search(r"(net_container|mlp)\.\d+\.weight", s) for s in ks):
                best = v
                break
    if best is not None:
        return best

    raise RuntimeError("Could not locate policy parameters in checkpoint")

# ----------- scan any object for running_mean / running_var -----------
def _scan_for_stats(obj, expected_dim=None):
    """
    Walk nested dict-like object and find (running_mean, running_var) or (mean, var).
    Return (rm, rv, where_string) or (None, None, None).
    """
    if not isinstance(obj, dict):
        return None, None, None

    # Collect candidate tensors and their paths
    found = []
    def walk(d, path=""):
        if isinstance(d, dict):
            for k, v in d.items():
                walk(v, f"{path}/{k}" if path else k)
        elif isinstance(d, torch.Tensor):
            found.append((path, v))  # (path, tensor)
    # NOTE: use v variable name correctly
    def walk(d, path=""):
        if isinstance(d, dict):
            for k, v in d.items():
                walk(v, f"{path}/{k}" if path else k)
        elif isinstance(d, torch.Tensor):
            found.append((path, d))
    walk(obj)

    # Filter names we care about
    mean_like = []
    var_like  = []
    for p, t in found:
        pl = p.lower()
        if any(x in pl for x in ["running_mean", "statistics/mean", "mean"]) and t.dim() == 1:
            mean_like.append((p, t))
        if any(x in pl for x in ["running_var", "running_variance", "statistics/var", "var"]) and t.dim() == 1:
            var_like.append((p, t))

    # pair by (parent path)
    def parent_of(s):
        return s.rsplit("/", 1)[0] if "/" in s else ""
    best = None
    best_score = -1e9
    for pm, tm in mean_like:
        for pv, tv in var_like:
            if parent_of(pm) != parent_of(pv):
                continue
            score = 0
            if tm.numel() == tv.numel(): score += 1
            if expected_dim and tm.numel() == expected_dim: score += 100
            if any(w in parent_of(pm) for w in ["obs","observation","normal","scaler","state_preprocessor"]): score += 5
            if score > best_score:
                best_score = score
                best = (tm, tv, f"{parent_of(pm)}")
    if best:
        rm, rv, where = best
        return _to_f32(rm), _to_f32(rv), where

    return None, None, None

# --------------- find running_mean / running_var ----------------------
def _find_running_stats_in_ckpt(raw, expected_dim=None):
    """
    Search nested dict for (running_mean, running_var) or (mean, var).
    Prefer vectors of length == expected_dim (e.g., 35).
    """
    # First, try generic scanner
    rm, rv, where = _scan_for_stats(raw, expected_dim)
    if rm is not None:
        print(f"🔎 Found stats in checkpoint @ {where}")
        return rm, rv

    # Legacy name-based pass
    candidates = []
    for k, v in _list_keys(raw):
        if isinstance(v, torch.Tensor):
            kl = k.lower().replace("\\", "/")
            if any(s in kl for s in ["running_mean", "running_var", "statistics/mean", "statistics/var", "state_preprocessor"]):
                candidates.append((kl, v))

    # group by parent path to find siblings
    by_parent = {}
    for k, v in candidates:
        parent = k.rsplit("/", 1)[0] if "/" in k else ""
        by_parent.setdefault(parent, {})[k.split("/")[-1]] = v

    best = None
    best_score = -1e9
    for parent, group in by_parent.items():
        rm = group.get("running_mean") or group.get("statistics/mean") or group.get("mean")
        rv = group.get("running_var")  or group.get("statistics/var")  or group.get("var")
        if isinstance(rm, torch.Tensor) and isinstance(rv, torch.Tensor):
            score = 0
            if rm.numel() == rv.numel(): score += 1
            if expected_dim and rm.numel() == expected_dim: score += 100
            if any(w in parent for w in ["obs", "observation", "normal", "scaler", "state_preprocessor"]): score += 5
            if score > best_score:
                best_score = score
                best = (rm, rv)

    if best:
        rm, rv = best
        return _to_f32(rm), _to_f32(rv)

    return None, None

def _load_stats_file(stats_path, expected_dim=None):
    """
    Load a .pt or .pkl and extract (running_mean, running_var), supporting nested dicts.
    """
    data = _try_load(stats_path)
    if not isinstance(data, dict):
        raise RuntimeError(f"Stats file '{stats_path}' is not a dict-like serialization")

    # Fast path: direct keys
    rm = data.get("running_mean", None) or data.get("mean", None)
    rv = data.get("running_var", None)  or data.get("running_variance", None) or data.get("var", None)
    if isinstance(rm, torch.Tensor) and isinstance(rv, torch.Tensor):
        return rm.float(), rv.float()

    # Nested path: scan
    rm, rv, where = _scan_for_stats(data, expected_dim)
    if rm is not None and rv is not None:
        print(f"🔎 Extracted nested stats from {stats_path} @ {where}")
        return rm, rv

    raise RuntimeError(f"Stats file '{stats_path}' does not contain running_mean/var")

def _auto_discover_stats(ckpt_path, expected_dim):
    """
    If stats are not in the checkpoint and --stats is not provided,
    scan the run directory (including 'params/') for .pt / .pkl files
    likely to contain the stats.
    """
    # run root: go up from /checkpoints/... to the run directory
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(ckpt_path)))
    print(f"🔎 Scanning '{run_dir}' for observation stats …")

    exts = (".pt", ".pkl")
    name_hints = ("agent", "env", "preproc", "obs", "state", "stats")
    for dp, _, fs in os.walk(run_dir):
        for f in fs:
            if not f.lower().endswith(exts):
                continue
            lname = f.lower()
            if not any(tag in lname for tag in name_hints):
                continue
            path = os.path.join(dp, f)
            try:
                obj = _try_load(path)
                rm, rv, where = _scan_for_stats(obj, expected_dim)
                if rm is not None:
                    rel = os.path.relpath(path, run_dir)
                    print(f"   ✅ found stats in {rel} @ {where}")
                    return rm, rv
            except Exception:
                # ignore unreadable files
                pass
    print("   ❌ no stats found in run directory.")
    return None, None

# ------------------ reconstruct MLP from shapes -----------------------
def _infer_mlp_layers(policy_state):
    """
    Detect sequential linear layers like 'net_container.0.weight', 'net_container.2.weight', ...
    Fall back to 'mlp.N.weight'. Returns [(in, out), ...] in order.
    """
    pat = re.compile(r"(net_container|mlp)\.(\d+)\.weight$")
    layers = []
    for k, v in policy_state.items():
        if isinstance(v, torch.Tensor):
            m = pat.search(k)
            if m:
                idx = int(m.group(2))
                layers.append((idx, v))
    if not layers:
        raise RuntimeError("Could not infer MLP layers (expected keys like 'net_container.N.weight').")

    layers.sort(key=lambda x: x[0])
    sizes = [(w.shape[1], w.shape[0]) for _, w in layers]  # (in, out)
    return sizes

def _find_action_head(policy_state):
    """
    Find the final linear mapping to actions.
    Try 'policy_layer.weight', else common aliases.
    """
    head_candidates = [
        "policy_layer.weight", "actor.mu.weight", "actor_mean.weight",
        "pi_mean.weight", "action_layer.weight", "actor.head.weight"
    ]
    for k in head_candidates:
        if k in policy_state:
            w = policy_state[k]
            if isinstance(w, torch.Tensor) and w.dim() == 2:
                return w.shape[1], w.shape[0], k  # (in, out, key)
    # fallback: guess by largest linear not in the trunk set
    linear_keys = [(k, v) for k, v in policy_state.items() if isinstance(v, torch.Tensor) and k.endswith(".weight") and v.dim() == 2]
    if not linear_keys:
        raise RuntimeError("No linear weights found for action head.")
    k, v = sorted(linear_keys, key=lambda kv: kv[1].shape[0])[-1]
    return v.shape[1], v.shape[0], k

# -------------------------- model ------------------------------------
class SkrlPolicyWithNorm(torch.nn.Module):
    def __init__(self, layer_sizes, head_in, head_out, running_mean=None, running_var=None):
        super().__init__()
        self.obs_dim = layer_sizes[0][0]
        self.action_dim = head_out

        # TorchScript-friendly epsilon
        self.register_buffer("eps", torch.tensor(1e-8, dtype=torch.float32))

        # Always register these so TorchScript sees attributes even without stats
        self.register_buffer("running_mean", torch.zeros(self.obs_dim, dtype=torch.float32))
        self.register_buffer("running_var",  torch.ones(self.obs_dim,  dtype=torch.float32))
        self.normalize: bool = False

        if running_mean is not None and running_var is not None:
            assert running_mean.numel() == self.obs_dim and running_var.numel() == self.obs_dim, \
                f"Stats dim {running_mean.numel()} != obs_dim {self.obs_dim}"
            with torch.no_grad():
                self.running_mean.copy_(running_mean.view(-1))
                self.running_var.copy_(running_var.view(-1))
            self.normalize = True

        layers = []
        for din, dout in layer_sizes:
            layers += [torch.nn.Linear(din, dout), torch.nn.ELU()]
        self.net_container = torch.nn.Sequential(*layers)

        # Final action head (name matches checkpoint key 'policy_layer.*')
        self.policy_layer = torch.nn.Linear(head_in, head_out)

    def forward(self, x):
        if self.normalize:
            x = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        h = self.net_container(x)
        return self.policy_layer(h)  # raw means; env handles scaling

# ------------------------ export --------------------------------------
def export(ckpt_path, out_path, stats_path=None, allow_no_stats=False):
    print(f"📦 Loading checkpoint: {ckpt_path}")
    raw = _try_load(ckpt_path)

    policy_state = _extract_policy_state(raw)

    # infer trunk and head sizes
    trunk_sizes = _infer_mlp_layers(policy_state)
    head_in, head_out, head_key = _find_action_head(policy_state)
    obs_dim = trunk_sizes[0][0]
    action_dim = head_out
    print(f"📊 Model dimensions: {obs_dim} obs → {action_dim} actions (head='{head_key}')")

    # 1) try inside checkpoint
    rm, rv = _find_running_stats_in_ckpt(raw, expected_dim=obs_dim)

    # 2) explicit --stats
    if (rm is None or rv is None) and stats_path:
        print(f"🔎 Using --stats: {stats_path}")
        rm, rv = _load_stats_file(stats_path, expected_dim=obs_dim)

    # 3) auto-discover in run directory (params/ etc.)
    if rm is None or rv is None:
        auto_rm, auto_rv = _auto_discover_stats(ckpt_path, expected_dim=obs_dim)
        if auto_rm is not None:
            rm, rv = auto_rm, auto_rv

    # Final decision
    if rm is None or rv is None:
        msg = ("No RunningStandardScaler stats found. Exporting WITHOUT normalization. "
               "Behavior may not match training.")
        if not allow_no_stats:
            raise RuntimeError(msg + " Pass --stats or use --allow-no-stats.")
        print(f"⚠️  WARNING: {msg}")
    else:
        _dump_stats(out_path, rm, rv)

    # build model
    model = SkrlPolicyWithNorm(trunk_sizes, head_in, head_out, rm, rv)

    # load weights (map matching keys only)
    own = model.state_dict()
    loaded = 0
    for k, v in policy_state.items():
        if k in own and own[k].shape == v.shape:
            own[k] = v
            loaded += 1
    model.load_state_dict(own, strict=False)
    print(f"🔧 Loaded {loaded} weight tensors into deploy model.")

    # small smoke test
    with torch.no_grad():
        dummy = torch.randn(1, obs_dim)
        out = model(dummy)
        assert out.shape == (1, action_dim)
        print(f"✅ Forward ok. Output shape: {out.shape}")

    # Prefer scripting
    scripted = torch.jit.script(model)
    torch.jit.save(scripted, out_path)
    print(f"✅ Exported TorchScript to: {out_path}")

# ------------------------ CLI -----------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Export TorchScript policy with embedded normalization")
    ap.add_argument("--ckpt", required=True, help="Path to SKRL checkpoint (.pt)")
    ap.add_argument("--out",  required=True, help="Output TorchScript path")
    ap.add_argument("--stats", default=None, help="Optional path to a file containing running_mean/var (.pt or .pkl)")
    ap.add_argument("--allow-no-stats", action="store_true",
                    help="Allow export without normalization (not recommended)")
    args = ap.parse_args()

    export(args.ckpt, args.out, args.stats, args.allow_no_stats)

if __name__ == "__main__":
    main()
