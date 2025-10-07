#!/usr/bin/env python3
import os, math, argparse, torch
import numpy as np

# Isaac Lab
import isaaclab.sim as sim_utils
from isaaclab.app import AppLauncher
from grab_skywalker.mdp.ptp_env_cfg import PTPEnvCfg_PLAY  # your uploaded config
# ^ prints "[PTPEnvCfg] Using ptp_env_cfg.py from: ..." on load

EPS = 1e-8

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="TorchScript policy (from fixed_export.py)")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--print_every", type=int, default=60)
    args, app_args = ap.parse_known_args()
    AppLauncher.add_app_launcher_args(ap)

    # launch sim
    launcher = AppLauncher(ap.parse_args([]))
    simulation_app = launcher.app

    # env (play config)
    cfg = PTPEnvCfg_PLAY()
    # - important bits we mirror from training:
    #   cfg.actions.arm_action.scale == 0.5
    #   cfg.actions.arm_action.use_default_offset == True
    #   cfg.num_observations == 35
    #   cfg.num_actions == 7

    # build env the Isaac-Lab way
    from isaaclab.envs import ManagerBasedRLEnv
    env: ManagerBasedRLEnv = ManagerBasedRLEnv(cfg)

    # model (+ embedded normalizer; your export wrote <model>_obs_stats.pt next to it)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy = torch.jit.load(args.model, map_location=device).eval().to(device)

    # convenience handles
    robot = env.scene["robot"]
    scale = 0.5  # from ActionsCfg
    steps = args.steps
    print_every = args.print_every

    # one env
    obs = env.reset()[0:1]  # (1, 35)

    for t in range(steps):
        # --- policy forward on current obs (policy includes normalization from export)
        with torch.no_grad():
            pi_out = policy(obs.to(device)).cpu()  # (1, 7) raw means
        # training-time action bounds: SKRL sends tanh-squashed or later clipped to [-1,1]
        a_clipped = torch.clamp(pi_out, -1.0, 1.0)  # (1,7)

        # env’s joint target mapping (this is what JointPositionActionCfg does with use_default_offset=True)
        q_def = robot.data.default_joint_pos[0:1, :7].cpu()
        q_cur = robot.data.joint_pos[0:1, :7].cpu()
        q_tgt = q_def + scale * a_clipped  # ABSOLUTE around defaults (not current!)

        # let Isaac Lab handle action application normally:
        # we pass the clipped action to the env; the action manager applies the same mapping internally.
        obs, rew, terminated, truncated, info = env.step(a_clipped)

        if t % print_every == 0:
            # joint limits (soft)
            lims = robot.data.soft_joint_pos_limits[0, :7].detach().cpu().numpy()  # [7,2]
            low, high = lims[:,0], lims[:,1]
            rng = high - low

            print(f"\nStep {t:4d}")
            print(f"raw π out (first): {pi_out[0].numpy().round(3)}")
            print(f"clipped action    : {a_clipped[0].numpy().round(3)}")
            print(f"scale, offset     : scale=0.5, use_default_offset=True")
            print(f"q_def             : {q_def[0].numpy().round(5)}")
            print(f"q_cur             : {q_cur[0].numpy().round(5)}")
            print(f"q_tgt (def+0.5*a) : {q_tgt[0].numpy().round(5)}")
            print(f"soft lims low     : {low.round(5)}")
            print(f"soft lims high    : {high.round(5)}")
            print(f"soft lims range   : {rng.round(5)}")

    simulation_app.close()

if __name__ == "__main__":
    main()
