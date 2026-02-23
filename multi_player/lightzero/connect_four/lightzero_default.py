import torch
import numpy as np
import gymnasium as gym
from easydict import EasyDict
from typing import Optional, Dict, Any, List
from functools import partial

# --- LightZero & Ding Imports ---
from lzero.entry import train_muzero
from lzero.policy.gumbel_muzero import GumbelMuZeroPolicy
from zoo.board_games.connect4.config.connect4_muzero_sp_mode_config import main_config, create_config
from zoo.board_games.connect4.envs.connect4_env import Connect4Env
from ding.utils import set_pkg_seed, ENV_REGISTRY
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.envs.env_manager import create_env_manager

# --- PettingZoo Import ---
from pettingzoo.classic import connect_four_v3


def train():
    # Detect GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    main_config.policy.cuda = True
    print(f"Using device: {device}")

    train_muzero([EasyDict(main_config), EasyDict(create_config)], seed=0)

if __name__ == "__main__":
    train()