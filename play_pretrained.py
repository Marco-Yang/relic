#!/usr/bin/env python3
"""
Modified play script to use pretrained TorchScript models directly.
This script bypasses RSL-RL's weight loading and uses the pretrained policy directly.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play with pretrained TorchScript policy.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during play."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default="Isaac-Spot-Interlimb-Play-v0", help="Name of the task.")
parser.add_argument(
    "--center", action="store_true", default=False, help="Look at the robot."
)
parser.add_argument(
    "--policy_path", 
    type=str, 
    default="source/relic/relic/assets/spot/pretrained/policy.pt",
    help="Path to the pretrained TorchScript policy."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import parse_env_cfg

# Import extensions to set up environment tasks
import relic.tasks  # noqa: F401


def main():
    """Play with pretrained TorchScript policy."""
    # parse environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    if args_cli.center:
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = (3.0, 3.0, 3.0)
        env_cfg.viewer.resolution = (4096, 2160)

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )
    
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(os.getcwd(), "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # load pretrained TorchScript policy
    policy_path = os.path.abspath(args_cli.policy_path)
    print(f"[INFO]: Loading TorchScript policy from: {policy_path}")
    
    try:
        policy = torch.jit.load(policy_path, map_location=env.unwrapped.device)
        print("[INFO]: Successfully loaded TorchScript policy")
    except Exception as e:
        print(f"[ERROR]: Failed to load policy: {e}")
        env.close()
        simulation_app.close()
        return

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    
    print("[INFO]: Starting simulation...")
    
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # extract policy observations (usually the first group)
            if isinstance(obs, dict):
                # Get the policy observations
                policy_obs = obs.get("policy", obs.get("observation", None))
                if policy_obs is None:
                    # Try to get the first available observation group
                    policy_obs = next(iter(obs.values()))
                obs_tensor = policy_obs.to(device=env.unwrapped.device, dtype=torch.float32)
            else:
                obs_tensor = obs.to(device=env.unwrapped.device, dtype=torch.float32)
            
            # get action from policy
            actions = policy(obs_tensor)
            
            # env stepping
            step_result = env.step(actions)
            if len(step_result) == 5:
                obs, _, _, _, info = step_result
            else:
                obs, _, _, info = step_result
            
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep >= args_cli.video_length:
                break

    # close the simulator
    env.close()
    print("[INFO]: Simulation completed successfully!")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()