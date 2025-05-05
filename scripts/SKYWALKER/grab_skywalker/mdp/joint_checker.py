# grab_skywalker/mdp/debug.py
import torch

def print_first_action(env):
    """MDP term that prints the first environment's action tensor."""
    a = env.action_manager.action            # (num_envs, action_dim)
    # convert to CPU numpy for neat printing
    print("[DEBUG] action[0] =", a[0].cpu().numpy())
