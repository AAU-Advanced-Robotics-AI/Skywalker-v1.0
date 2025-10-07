#!/usr/bin/env python3

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a trained agent checkpoint for the SKYWALKER grab environment.
This script directly loads the environment without relying on complex parsing.
"""

import argparse
import torch
import numpy as np
from typing import Dict, Any

# IsaacLab imports
from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Play trained SKYWALKER grab agent")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=42, help="Seed for environment.")

# Append AppLauncher CLI arguments
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.dict import print_dict

# Import your environment configurations
import grab_skywalker
from grab_skywalker.config.xarm7.ptp_env_cfg import PTPEnvCfg_PLAY

# Try to import skrl for loading the agent
try:
    import skrl
    from skrl.agents.torch.ppo import PPO as PPO_SKRL
    from skrl.memories.torch import RandomMemory
    from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
    from skrl.utils.model_instantiators.torch import Shape
    SKRL_AVAILABLE = True
    print("[INFO] SKRL available - will load trained agent")
except ImportError:
    SKRL_AVAILABLE = False
    print("[WARNING] SKRL not available - will use random actions")


class DummyPolicy:
    """Dummy policy for when SKRL is not available."""
    
    def __init__(self, action_space):
        self.action_space = action_space
        
    def act(self, observations, deterministic=True):
        batch_size = observations.shape[0] if hasattr(observations, 'shape') else 1
        return torch.randn(batch_size, *self.action_space.shape)


def load_skrl_agent(checkpoint_path: str, env: ManagerBasedRLEnv) -> Any:
    """Load a trained SKRL PPO agent from checkpoint."""
    
    if not SKRL_AVAILABLE:
        print("[WARNING] SKRL not available, using dummy policy")
        return DummyPolicy(env.action_space)
    
    # Define models for SKRL PPO
    class Policy(GaussianMixin, Model):
        def __init__(self, observation_space, action_space, device, **kwargs):
            Model.__init__(self, observation_space, action_space, device, **kwargs)
            GaussianMixin.__init__(self, clip_actions=True, clip_log_std=True, min_log_std=-20, max_log_std=2)

            self.net = torch.nn.Sequential(
                torch.nn.Linear(self.num_observations, 256),
                torch.nn.ELU(),
                torch.nn.Linear(256, 128),
                torch.nn.ELU(),
                torch.nn.Linear(128, 64),
                torch.nn.ELU()
            )
            
            self.mean_layer = torch.nn.Linear(64, self.num_actions)
            self.log_std_parameter = torch.nn.Parameter(torch.zeros(self.num_actions))

        def compute(self, inputs, role):
            x = self.net(inputs["states"])
            return self.mean_layer(x), self.log_std_parameter, {}

    class Value(DeterministicMixin, Model):
        def __init__(self, observation_space, action_space, device, **kwargs):
            Model.__init__(self, observation_space, action_space, device, **kwargs)
            DeterministicMixin.__init__(self)

            self.net = torch.nn.Sequential(
                torch.nn.Linear(self.num_observations, 256),
                torch.nn.ELU(),
                torch.nn.Linear(256, 128),
                torch.nn.ELU(),
                torch.nn.Linear(128, 64),
                torch.nn.ELU(),
                torch.nn.Linear(64, 1)
            )

        def compute(self, inputs, role):
            return self.net(inputs["states"]), {}

    # Create device
    device = env.device

    # Create models
    models = {}
    models["policy"] = Policy(env.observation_space, env.action_space, device)
    models["value"] = Value(env.observation_space, env.action_space, device)

    # Create memory (not used during inference)
    memory = RandomMemory(memory_size=1, num_envs=env.num_envs, device=device)

    # Create PPO agent
    agent = PPO_SKRL(
        models=models,
        memory=memory,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    # Load checkpoint
    try:
        agent.load(checkpoint_path)
        print(f"[INFO] Successfully loaded checkpoint: {checkpoint_path}")
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint: {e}")
        return DummyPolicy(env.action_space)

    return agent


def main():
    """Play with the trained agent."""
    
    # Create environment configuration
    env_cfg = PTPEnvCfg_PLAY()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device

    # Create environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    print(f"[INFO] Created environment with {env.num_envs} environment(s)")
    print(f"[INFO] Action space: {env.action_space}")
    print(f"[INFO] Observation space: {env.observation_space}")

    # Load the trained agent
    agent = load_skrl_agent(args.checkpoint, env)

    # Reset environment
    obs, _ = env.reset(seed=args.seed)
    
    print("[INFO] Starting simulation...")
    print("[INFO] Press 'Ctrl+C' to stop")

    # Simulation loop
    step_count = 0
    episode_rewards = torch.zeros(env.num_envs, device=env.device)
    
    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                # Get actions from agent
                if SKRL_AVAILABLE and hasattr(agent, 'act'):
                    # SKRL agent
                    actions = agent.act(obs["policy"], deterministic=True)[0]
                else:
                    # Dummy policy
                    actions = agent.act(obs["policy"], deterministic=True)
                
                # Step environment
                obs, rewards, terminations, truncations, info = env.step(actions)
                
                # Accumulate rewards
                episode_rewards += rewards
                
                # Handle resets
                reset_mask = terminations | truncations
                if reset_mask.any():
                    reset_envs = torch.where(reset_mask)[0]
                    print(f"[INFO] Episode(s) finished in env(s): {reset_envs.tolist()}")
                    print(f"[INFO] Episode reward(s): {episode_rewards[reset_envs].tolist()}")
                    episode_rewards[reset_mask] = 0.0
                
                step_count += 1
                
                # Print progress every 100 steps
                if step_count % 100 == 0:
                    print(f"[INFO] Steps: {step_count}, Mean reward so far: {episode_rewards.mean().item():.3f}")

    except KeyboardInterrupt:
        print("\n[INFO] Simulation stopped by user")
    
    finally:
        # Close environment
        env.close()
        print("[INFO] Environment closed")


if __name__ == "__main__":
    main()
    simulation_app.close()
