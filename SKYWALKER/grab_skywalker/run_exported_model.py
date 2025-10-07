#!/usr/bin/env python3

"""
Test the exported SKYWALKER PTP model in Isaac Sim environment.
This script loads the exported PyTorch JIT model and runs it in the actual simulation
to verify that the physics and behavior translate correctly.
"""

import argparse
import torch
import numpy as np
import time

# Parse command line arguments
parser = argparse.ArgumentParser(description="Run exported SKYWALKER model in Isaac Sim")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--model_path", type=str, 
                   default="./exported_models/skywalker_ptp_policy.pt",
                   help="Path to exported model")
parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
parser.add_argument("--max_steps", type=int, default=500, help="Max steps per episode")
parser.add_argument("--seed", type=int, default=42, help="Seed for environment")

# Import Isaac Lab modules
from isaaclab.app import AppLauncher

# Append AppLauncher CLI arguments
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import Isaac Lab modules after launching
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.dict import print_dict

# Import our environment
import grab_skywalker
from grab_skywalker.config.xarm7.ptp_env_cfg import PTPEnvCfg_PLAY

def load_exported_model(model_path, device):
    """Load the exported PyTorch JIT model."""
    print(f"🔄 Loading exported model from: {model_path}")
    try:
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
        print(f"✅ Model loaded successfully on {device}")
        return model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

def run_model_in_sim():
    """Run the exported model in Isaac Sim environment."""
    
    # Load the exported model
    model = load_exported_model(args_cli.model_path, app_launcher.device)
    if model is None:
        return
    
    # Create environment configuration
    env_cfg = PTPEnvCfg_PLAY()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    print(f"🌍 Creating environment with {args_cli.num_envs} environments...")
    
    # Create the environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    print("📊 Environment Info:")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")
    print(f"   Number of environments: {env.num_envs}")
    print(f"   Max episode length: {env.max_episode_length}")
    
    # Statistics tracking
    episode_rewards = []
    episode_lengths = []
    total_steps = 0
    inference_times = []
    
    print(f"\n🚀 Starting {args_cli.episodes} episodes...")
    print("=" * 60)
    
    for episode in range(args_cli.episodes):
        print(f"\n📺 Episode {episode + 1}/{args_cli.episodes}")
        
        # Reset environment
        obs, _ = env.reset()
        episode_reward = 0.0
        episode_length = 0
        
        for step in range(args_cli.max_steps):
            # Convert observations to model input format
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=app_launcher.device)
            
            # Run inference
            start_time = time.time()
            with torch.no_grad():
                actions = model(obs_tensor)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Convert actions back to numpy for environment
            actions_np = actions.cpu().numpy()
            
            # Step environment
            obs, rewards, terminated, truncated, infos = env.step(actions_np)
            
            # Update statistics
            episode_reward += rewards.mean().item()
            episode_length += 1
            total_steps += 1
            
            # Print progress every 100 steps
            if step % 100 == 0:
                avg_reward = episode_reward / episode_length
                avg_inference = np.mean(inference_times[-100:]) * 1000  # ms
                print(f"   Step {step:3d}: Reward={avg_reward:+.3f}, "
                      f"Inference={avg_inference:.2f}ms")
            
            # Check if episode is done
            if terminated.any() or truncated.any():
                break
        
        # Episode summary
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"   ✅ Episode {episode + 1} complete:")
        print(f"      Total reward: {episode_reward:+.3f}")
        print(f"      Episode length: {episode_length} steps")
        print(f"      Avg reward/step: {episode_reward/episode_length:+.4f}")
    
    # Final statistics
    print("\n" + "=" * 60)
    print("🏁 SIMULATION COMPLETE")
    print("=" * 60)
    
    print(f"📊 Performance Statistics:")
    print(f"   Total episodes: {len(episode_rewards)}")
    print(f"   Total steps: {total_steps}")
    print(f"   Avg episode reward: {np.mean(episode_rewards):+.3f} ± {np.std(episode_rewards):.3f}")
    print(f"   Avg episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"   Max episode reward: {np.max(episode_rewards):+.3f}")
    print(f"   Min episode reward: {np.min(episode_rewards):+.3f}")
    
    print(f"\n⚡ Inference Performance:")
    avg_inference = np.mean(inference_times) * 1000
    std_inference = np.std(inference_times) * 1000
    max_inference = np.max(inference_times) * 1000
    fps = 1.0 / np.mean(inference_times)
    print(f"   Avg inference time: {avg_inference:.2f} ± {std_inference:.2f} ms")
    print(f"   Max inference time: {max_inference:.2f} ms")
    print(f"   Inference FPS: {fps:.0f} Hz")
    
    # Check if performance is good
    success_rate = sum(1 for r in episode_rewards if r > 0) / len(episode_rewards)
    print(f"\n🎯 Success Analysis:")
    print(f"   Episodes with positive reward: {success_rate:.1%}")
    
    if success_rate > 0.5:
        print("   ✅ Model appears to be working well!")
    elif success_rate > 0.2:
        print("   ⚠️  Model shows some success but may need tuning")
    else:
        print("   ❌ Model may need retraining or debugging")
    
    # Close environment
    env.close()

if __name__ == "__main__":
    try:
        run_model_in_sim()
    except Exception as e:
        print(f"❌ Error during simulation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close simulation
        simulation_app.close()
