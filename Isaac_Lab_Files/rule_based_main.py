# Python scripts/reinforcement_learning/sb3/train.py 
# --task=Foosball-1player-v0 
# --headless  
# --video  
# --video_length 500  
# --video_interval 10000


"""---------------------------------"""
"""Launch Isaac Sim Simulator first."""

import argparse
import contextlib
import signal
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--video", action="store_true", default=True, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=10000, help="Interval between video recordings (in steps).")
parser.add_argument("--task", type=str, default="Foosball-1player-v0", help="Name of the task.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
args_cli.enable_cameras = True
args_cli.headless = True
# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

def cleanup_pbar(*args):
    """
    A small helper to stop training and
    cleanup progress bar properly on ctrl+c
    """
    import gc

    tqdm_objects = [obj for obj in gc.get_objects() if "tqdm" in type(obj).__name__]
    for tqdm_object in tqdm_objects:
        if "tqdm_rich" in type(tqdm_object).__name__:
            tqdm_object.close()
    raise KeyboardInterrupt
# disable KeyboardInterrupt override
signal.signal(signal.SIGINT, cleanup_pbar)


"""---------------------------------"""
"""other imports"""
import gymnasium as gym
import logging
import numpy as np
import os
import random
from datetime import datetime

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

from rule_based_foosball_agent import *
from foosball2.foosball_env import *

"""---------------------------------"""
"""main"""

def main():
    # set up logging and output etc. if needed 
    log_dir = os.path.join(os.getcwd(), f"isaac_log_{datetime.now().date()}")

    # set up env 
    env_cfg = FoosballEnvCfg()
    env = gym.make("Foosball-1player-v0", cfg=env_cfg, render_mode="rgb_array")

    # sets up video recording
    
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    agent = RuleBasedFoosballAgent()

    max_step = 10
    obs, info = env.reset(seed=42)
    for step in range(max_step):

        # action shape is (num_envs, action_dim)
        action = torch.from_numpy(np.zeros((1, 8), dtype = np.float32))
        env.step(action)
        # # Compute deterministic action
        # action = agent.compute_action(obs)

        # # Step the environment
        # obs, reward, done, trunc, info = env.step(action)


    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
