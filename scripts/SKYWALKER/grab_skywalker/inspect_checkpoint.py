#!/usr/bin/env python3

"""
Inspect SKRL checkpoint structure to understand the model format.
"""

import torch
import argparse

def inspect_checkpoint(checkpoint_path: str):
    """Inspect the structure of a SKRL checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print("\n=== CHECKPOINT STRUCTURE ===")
    
    def print_nested_dict(d, indent=0):
        for key, value in d.items():
            prefix = "  " * indent
            if isinstance(value, dict):
                print(f"{prefix}{key}:")
                print_nested_dict(value, indent + 1)
            elif isinstance(value, torch.Tensor):
                print(f"{prefix}{key}: Tensor{list(value.shape)} ({value.dtype})")
            elif hasattr(value, '__dict__'):
                print(f"{prefix}{key}: {type(value).__name__} object")
                if hasattr(value, 'state_dict'):
                    print(f"{prefix}  -> has state_dict with keys: {list(value.state_dict().keys())[:5]}...")
            else:
                print(f"{prefix}{key}: {type(value).__name__}")
    
    print_nested_dict(checkpoint)
    
    # Try to extract the policy network
    print("\n=== LOOKING FOR POLICY NETWORK ===")
    
    if 'policy' in checkpoint:
        policy = checkpoint['policy']
        print(f"Policy type: {type(policy)}")
        
        if hasattr(policy, 'state_dict'):
            policy_state = policy.state_dict()
            print(f"Policy state_dict keys: {list(policy_state.keys())}")
            
            # Find input/output dimensions
            for key, tensor in policy_state.items():
                if 'weight' in key:
                    print(f"  {key}: {list(tensor.shape)}")
        elif isinstance(policy, dict):
            print(f"Policy dict keys: {list(policy.keys())}")
            for key, value in policy.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: Tensor{list(value.shape)}")

def main():
    parser = argparse.ArgumentParser(description="Inspect SKRL checkpoint structure")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    
    args = parser.parse_args()
    inspect_checkpoint(args.checkpoint)

if __name__ == "__main__":
    main()
