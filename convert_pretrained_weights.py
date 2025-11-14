#!/usr/bin/env python3
"""
Script to convert TorchScript pretrained weights to standard PyTorch state dict format
compatible with RSL-RL training framework.
"""

import torch
import os

def convert_torchscript_to_state_dict(input_path: str, output_path: str):
    """
    Convert TorchScript model to standard PyTorch state dict format.
    
    Args:
        input_path: Path to the TorchScript model (.pt file)
        output_path: Path to save the converted state dict (.pt file)
    """
    print(f"Loading TorchScript model from: {input_path}")
    
    # Load the TorchScript model
    try:
        jit_model = torch.jit.load(input_path, map_location='cpu')
        print("Successfully loaded TorchScript model")
    except Exception as e:
        print(f"Error loading TorchScript model: {e}")
        return False
    
    # Extract state dict from TorchScript model
    try:
        # Try to get state dict directly
        if hasattr(jit_model, 'state_dict'):
            state_dict = jit_model.state_dict()
        else:
            # If direct state dict access fails, try to extract parameters
            state_dict = {}
            for name, param in jit_model.named_parameters():
                state_dict[name] = param.data
            for name, buffer in jit_model.named_buffers():
                state_dict[name] = buffer.data
        
        print(f"Extracted state dict with {len(state_dict)} parameters/buffers")
        
        # Create a mock training state dict format that RSL-RL expects
        training_state = {
            'model_state_dict': state_dict,
            'optimizer_state_dict': None,  # No optimizer state available
            'epoch': 10000,  # Assume final epoch
            'it': 10000,     # Assume final iteration
        }
        
        # Save the converted state dict
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(training_state, output_path)
        print(f"Successfully saved converted weights to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error extracting state dict: {e}")
        return False

def inspect_pretrained_model(model_path: str):
    """
    Inspect the structure of the pretrained model.
    """
    print(f"Inspecting model: {model_path}")
    
    try:
        # Try loading as TorchScript
        jit_model = torch.jit.load(model_path, map_location='cpu')
        print("Loaded as TorchScript model")
        
        # Print model structure
        print("\nModel structure:")
        print(jit_model)
        
        # Try to get parameters
        print("\nParameters:")
        for name, param in jit_model.named_parameters():
            print(f"  {name}: {param.shape}")
            
        print("\nBuffers:")
        for name, buffer in jit_model.named_buffers():
            print(f"  {name}: {buffer.shape}")
            
    except Exception as e:
        print(f"Error loading as TorchScript: {e}")
        
        # Try loading as regular PyTorch model
        try:
            model_dict = torch.load(model_path, map_location='cpu')
            print("Loaded as regular PyTorch model")
            print("Keys in model dict:")
            for key in model_dict.keys():
                print(f"  {key}")
                
        except Exception as e2:
            print(f"Error loading as regular PyTorch: {e2}")

def main():
    # Define paths
    input_path = "source/relic/relic/assets/spot/pretrained/policy.pt"
    output_path = "logs/rsl_rl/spot_interlimb/2024-11-14_15-00-00_pretrained/model_10000.pt"
    
    # First, inspect the pretrained model
    inspect_pretrained_model(input_path)
    
    print("\n" + "="*50)
    print("ALTERNATIVE SOLUTIONS:")
    print("="*50)
    print("Since the pretrained weights format is incompatible with RSL-RL,")
    print("here are alternative ways to run the play script:")
    print()
    print("1. Use the pretrained weights directly with custom loading:")
    print("   Modify the play script to load TorchScript models directly")
    print()
    print("2. Use command line parameters to specify a different checkpoint:")
    print(f"   python scripts/rsl_rl/play.py --task Isaac-Spot-Interlimb-Play-v0 \\")
    print(f"          --load_run 2024-11-14_15-00-00_pretrained \\")
    print(f"          --checkpoint model_10000.pt --center")
    print()
    print("3. Train a new model from scratch:")
    print("   python scripts/rsl_rl/train.py --task Isaac-Spot-Interlimb-Phase-1-v0 --headless")
    print()
    print("The pretrained policy.pt and policy.onnx files are likely meant for")
    print("deployment/inference rather than training continuation.")

if __name__ == "__main__":
    main()