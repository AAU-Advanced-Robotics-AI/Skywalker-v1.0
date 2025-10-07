#!/usr/bin/env python3
"""
Export script for Base Move agent with observation normalization embedding.

This script takes a trained base_move agent checkpoint and exports it as a TorchScript model
with embedded observation normalization, similar to fixed_export.py but for the base_move task.

The base_move agent uses a 35-D observation vector:
- joint_pos (7): relative joint positions
- joint_vel (7): relative joint velocities  
- base_pos_rel_ee (3): base position relative to end-effector
- base_quat_rel_ee (4): base orientation quaternion relative to end-effector
- target_pose_command (7): target pose command (position + quaternion)
- actions (7): last actions
"""

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
        if k in raw:
            candidate = raw[k]
            if isinstance(candidate, dict) and any("weight" in sk for sk in candidate.keys()):
                print(f"✅ Found policy state_dict at key: '{k}'")
                return candidate

    # 2) Dive into 'state_dict' if it exists
    if "state_dict" in raw:
        sd = raw["state_dict"]
        for k in direct_candidates:
            if k in sd:
                candidate = sd[k]
                if isinstance(candidate, dict) and any("weight" in sk for sk in candidate.keys()):
                    print(f"✅ Found policy state_dict at key: 'state_dict.{k}'")
                    return candidate

    # 3) Look for actor/policy prefixes in flat dict
    if isinstance(raw, dict):
        for prefix in ["policy", "actor", "pi"]:
            matches = {k: v for k, v in raw.items() if k.startswith(f"{prefix}.")}
            if matches:
                print(f"✅ Found policy state_dict with prefix: '{prefix}.'")
                return matches

    # 4) Recursive search in nested dicts
    for key, value in _list_keys(raw):
        if isinstance(value, dict) and any("weight" in k for k in value.keys()):
            if any(term in key.lower() for term in ["policy", "actor", "pi"]):
                print(f"✅ Found policy state_dict at nested key: '{key}'")
                return value

    raise ValueError("❌ Could not locate policy state_dict in checkpoint")

# -------------------- extract obs normalization stats ------------------
def _extract_obs_stats(raw):
    """
    Extract running_mean and running_var for observation normalization.
    Returns (running_mean, running_var) as torch tensors or None if not found.
    """
    # SKRL-specific locations first
    skrl_candidates = [
        ("preprocessors", "running_mean", "running_var"),
        ("observation_preprocessor", "running_mean", "running_var"),
        ("obs_preprocessor", "running_mean", "running_var"),
    ]
    
    # Common locations for obs stats
    stat_candidates = [
        ("obs_normalizer", "running_mean", "running_var"),
        ("obs_running_mean", "obs_running_var", None),
        ("running_mean", "running_var", None),
        ("observation_normalizer.running_mean", "observation_normalizer.running_var", None),
    ]
    
    all_candidates = skrl_candidates + stat_candidates

    for candidate in all_candidates:
        if len(candidate) == 3 and candidate[2] is None:
            # Two separate keys
            mean_key, var_key = candidate[0], candidate[1]
            if mean_key in raw and var_key in raw:
                rm = _to_f32(raw[mean_key])
                rv = _to_f32(raw[var_key])
                print(f"✅ Found obs stats: '{mean_key}', '{var_key}' -> shapes {rm.shape}, {rv.shape}")
                return rm, rv
        else:
            # Nested under common parent
            parent_key, mean_sub, var_sub = candidate
            if parent_key in raw:
                parent = raw[parent_key]
                if isinstance(parent, dict) and mean_sub in parent and var_sub in parent:
                    rm = _to_f32(parent[mean_sub])
                    rv = _to_f32(parent[var_sub])
                    print(f"✅ Found obs stats: '{parent_key}.{mean_sub}', '{parent_key}.{var_sub}' -> shapes {rm.shape}, {rv.shape}")
                    return rm, rv

    # Search in state_dict if it exists
    if "state_dict" in raw:
        sd = raw["state_dict"]
        for candidate in all_candidates:
            if len(candidate) == 3 and candidate[2] is None:
                mean_key, var_key = candidate[0], candidate[1]
                if mean_key in sd and var_key in sd:
                    rm = _to_f32(sd[mean_key])
                    rv = _to_f32(sd[var_key])
                    print(f"✅ Found obs stats in state_dict: '{mean_key}', '{var_key}' -> shapes {rm.shape}, {rv.shape}")
                    return rm, rv

    # Recursive search for stats
    all_keys = list(_list_keys(raw))
    print(f"🔍 All keys in checkpoint: {[k for k, _ in all_keys[:20]]}...")  # Show first 20 keys
    
    mean_keys = [k for k, _ in all_keys if "mean" in k.lower() and ("obs" in k.lower() or "preprocess" in k.lower())]
    var_keys = [k for k, _ in all_keys if "var" in k.lower() and ("obs" in k.lower() or "preprocess" in k.lower())]
    
    print(f"🔍 Found mean-like keys: {mean_keys}")
    print(f"🔍 Found var-like keys: {var_keys}")
    
    if mean_keys and var_keys:
        # Try to pair them
        for mk in mean_keys:
            for vk in var_keys:
                try:
                    mean_val = raw
                    var_val = raw
                    for part in mk.split('.'):
                        mean_val = mean_val[part]
                    for part in vk.split('.'):
                        var_val = var_val[part]
                    
                    rm = _to_f32(mean_val)
                    rv = _to_f32(var_val)
                    if rm.shape == rv.shape:
                        print(f"✅ Found obs stats via search: '{mk}', '{vk}' -> shape {rm.shape}")
                        return rm, rv
                except:
                    continue

    print("⚠️ No observation normalization stats found - will use zeros")
    return None, None

# -------------------- build export model ------------------
class BaseMoveExportModel(torch.nn.Module):
    """
    Base Move policy with embedded observation normalization.
    
    Forward pass:
    1. Normalize 35-D observation vector using running stats
    2. Pass through policy network
    3. Return 7-D action
    """
    
    def __init__(self, policy_state_dict, obs_mean=None, obs_var=None):
        super().__init__()
        
        # Build policy network from state dict
        self.policy = self._build_policy_from_state_dict(policy_state_dict)
        
        # Observation normalization stats (35-D for base_move)
        if obs_mean is not None and obs_var is not None:
            self.register_buffer("obs_mean", obs_mean.reshape(1, -1))
            self.register_buffer("obs_std", torch.sqrt(obs_var + EPS).reshape(1, -1))
            self.normalize_obs = True
        else:
            self.register_buffer("obs_mean", torch.zeros(1, 35))
            self.register_buffer("obs_std", torch.ones(1, 35))
            self.normalize_obs = False
            
        print(f"📏 Obs normalization: {'enabled' if self.normalize_obs else 'disabled'}")
        print(f"📏 Expected obs shape: {self.obs_mean.shape}")
        
    def _build_policy_from_state_dict(self, state_dict):
        """
        Infer network architecture from state_dict and build the policy.
        Handles SKRL policy structure with net_container and policy_layer.
        """
        print(f"🔍 Policy state dict keys: {list(state_dict.keys())}")
        
        # Check if this is SKRL format with net_container and policy_layer
        net_container_keys = [k for k in state_dict.keys() if k.startswith('net_container.')]
        policy_layer_keys = [k for k in state_dict.keys() if k.startswith('policy_layer.')]
        
        if net_container_keys and policy_layer_keys:
            print("🏗️ Detected SKRL policy structure (net_container + policy_layer)")
            return self._build_skrl_policy(state_dict)
        else:
            print("🏗️ Detected standard sequential structure")
            return self._build_sequential_policy(state_dict)
    
    def _build_skrl_policy(self, state_dict):
        """Build SKRL-style policy with net_container (shared layers) + policy_layer (output)."""
        
        # Extract net_container layers (shared feature extractor)
        net_container_keys = sorted([k for k in state_dict.keys() if k.startswith('net_container.')])
        
        # Group by layer index
        layer_groups = {}
        for key in net_container_keys:
            parts = key.split('.')
            if len(parts) >= 3:
                layer_idx = int(parts[1])
                param_type = parts[2]  # 'weight' or 'bias'
                if layer_idx not in layer_groups:
                    layer_groups[layer_idx] = {}
                layer_groups[layer_idx][param_type] = state_dict[key]
        
        # Build net_container layers
        layers = []
        layer_indices = sorted(layer_groups.keys())
        
        for i, layer_idx in enumerate(layer_indices):
            layer_params = layer_groups[layer_idx]
            if 'weight' in layer_params:
                weight = layer_params['weight']
                bias = layer_params.get('bias')
                
                out_dim, in_dim = weight.shape
                linear = torch.nn.Linear(in_dim, out_dim)
                linear.weight.data.copy_(weight)
                if bias is not None:
                    linear.bias.data.copy_(bias)
                
                layers.append(linear)
                
                # Add ELU between layers (but not after the last net_container layer)
                if i < len(layer_indices) - 1:
                    layers.append(torch.nn.ELU())
        
        # Add final policy layer
        if 'policy_layer.weight' in state_dict:
            policy_weight = state_dict['policy_layer.weight']
            policy_bias = state_dict.get('policy_layer.bias')
            
            out_dim, in_dim = policy_weight.shape
            policy_linear = torch.nn.Linear(in_dim, out_dim)
            policy_linear.weight.data.copy_(policy_weight)
            if policy_bias is not None:
                policy_linear.bias.data.copy_(policy_bias)
            
            # Add ELU before final layer if we have net_container layers
            if layers:
                layers.append(torch.nn.ELU())
            layers.append(policy_linear)
        
        net = torch.nn.Sequential(*layers)
        print(f"✅ Built SKRL policy: {len(layer_indices)} net_container layers + 1 policy layer")
        return net
    
    def _build_sequential_policy(self, state_dict):
        """Build standard sequential policy from numbered layer keys."""
        # Find layer dimensions from weight shapes
        layer_dims = []
        layer_keys = sorted([k for k in state_dict.keys() if k.endswith('.weight')])
        
        for key in layer_keys:
            weight = state_dict[key]
            if len(weight.shape) == 2:  # Linear layer
                out_dim, in_dim = weight.shape
                if not layer_dims:
                    layer_dims.append(in_dim)  # Input dimension
                layer_dims.append(out_dim)
        
        print(f"🏗️ Inferred network architecture: {layer_dims}")
        
        # Build network
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(torch.nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:  # No activation on final layer
                layers.append(torch.nn.ELU())
        
        net = torch.nn.Sequential(*layers)
        
        # Load the weights
        net.load_state_dict(state_dict, strict=True)
        print(f"✅ Built policy network with {len(layer_keys)} layers")
        
        return net
    
    def forward(self, obs):
        """
        Forward pass: normalize obs, then policy.
        
        Args:
            obs: (B, 35) observation tensor
            
        Returns:
            actions: (B, 7) action tensor
        """
        # Ensure correct shape
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        # Normalize observations
        if self.normalize_obs:
            obs_norm = (obs - self.obs_mean) / self.obs_std
        else:
            obs_norm = obs
            
        # Policy forward
        actions = self.policy(obs_norm)
        
        return actions

# -------------------- main export function ------------------
def export_base_move_policy(checkpoint_path, output_path):
    """Export base_move policy with embedded normalization."""
    
    print(f"🔄 Loading checkpoint: {checkpoint_path}")
    raw_ckpt = _try_load(checkpoint_path)
    
    # Extract policy state dict
    policy_state = _extract_policy_state(raw_ckpt)
    
    # Extract observation normalization stats
    obs_mean, obs_var = _extract_obs_stats(raw_ckpt)
    
    # Build export model
    export_model = BaseMoveExportModel(policy_state, obs_mean, obs_var)
    export_model.eval()
    
    # Test with dummy input
    print("🧪 Testing export model...")
    dummy_obs = torch.randn(1, 35)  # Base move uses 35-D observations
    with torch.no_grad():
        dummy_action = export_model(dummy_obs)
    print(f"✅ Test passed: obs {dummy_obs.shape} -> action {dummy_action.shape}")
    
    # Export to TorchScript
    print(f"📦 Exporting to TorchScript: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    scripted = torch.jit.script(export_model)
    scripted.save(output_path)
    
    # Save stats sidecar
    if obs_mean is not None and obs_var is not None:
        _dump_stats(output_path, obs_mean, obs_var)
    
    print(f"✅ Export complete: {output_path}")
    return output_path

# -------------------- CLI ------------------
def main():
    parser = argparse.ArgumentParser(description="Export Base Move policy with embedded normalization")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True,
        help="Path to trained base_move checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="scripts/SKYWALKER/grab_skywalker/exported_models/base_move_policy.pt",
        help="Output path for exported TorchScript model"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    try:
        export_base_move_policy(args.checkpoint, args.output)
        print(f"🎉 Successfully exported base_move policy!")
        print(f"📁 Model: {args.output}")
        print(f"📁 Stats: {os.path.splitext(args.output)[0]}_obs_stats.pt")
    except Exception as e:
        print(f"❌ Export failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
