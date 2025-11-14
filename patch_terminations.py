#!/usr/bin/env python3
"""
Patch for fixing single environment dimension issues in terminations.py
"""

import shutil
import os

def patch_terminations_file():
    """
    Create a backup and patch the terminations.py file to fix single environment issues.
    """
    terminations_path = "source/relic/relic/tasks/loco_manipulation/mdp/terminations.py"
    backup_path = terminations_path + ".backup"
    
    # Create backup if it doesn't exist
    if not os.path.exists(backup_path):
        shutil.copy2(terminations_path, backup_path)
        print(f"Created backup: {backup_path}")
    
    # Read the original file
    with open(terminations_path, 'r') as f:
        content = f.read()
    
    # Apply the patch
    old_code = """    # check if any contact force with the ground exceeds the threshold
    return (
        torch.max(torch.norm(contact_forces_with_ground, dim=-1), dim=1)[0] > threshold
    )"""
    
    new_code = """    # check if any contact force with the ground exceeds the threshold
    contact_norms = torch.norm(contact_forces_with_ground, dim=-1)
    
    # Handle both single and multi-environment cases
    if contact_norms.dim() == 1:
        # Single environment case
        max_contact_force = torch.max(contact_norms)
    else:
        # Multi-environment case
        max_contact_force = torch.max(contact_norms, dim=1)[0]
    
    return max_contact_force > threshold"""
    
    # Replace the problematic code
    if old_code in content:
        content = content.replace(old_code, new_code)
        
        # Write the patched file
        with open(terminations_path, 'w') as f:
            f.write(content)
        
        print("Successfully patched terminations.py")
        print("The file now handles both single and multi-environment cases.")
        return True
    else:
        print("Could not find the target code to patch.")
        return False

def restore_backup():
    """
    Restore the original file from backup.
    """
    terminations_path = "source/relic/relic/tasks/loco_manipulation/mdp/terminations.py"
    backup_path = terminations_path + ".backup"
    
    if os.path.exists(backup_path):
        shutil.copy2(backup_path, terminations_path)
        print("Restored original file from backup.")
        return True
    else:
        print("No backup file found.")
        return False

if __name__ == "__main__":
    print("Patching terminations.py for single environment compatibility...")
    success = patch_terminations_file()
    
    if success:
        print("\n✅ Patch applied successfully!")
        print("You can now run with single environment:")
        print("pixi run python play_pretrained.py --task Isaac-Spot-Interlimb-Play-v0 --center --num_envs 1")
        print("\nTo restore the original file, run:")
        print("python patch_terminations.py restore")
    else:
        print("\n❌ Patch failed. Please check the file manually.")