import torch
import numpy as np
import gymnasium as gym
from easydict import EasyDict
from typing import Optional, Dict, Any, List
from functools import partial

# --- LightZero & Ding Imports ---
from lzero.entry import train_muzero
# from lzero.policy.gumbel_muzero import GumbelMuZeroPolicy
from zoo.board_games.tictactoe.config.tictactoe_muzero_sp_mode_config import main_config as sp_main_config
from zoo.board_games.tictactoe.config.tictactoe_muzero_sp_mode_config import create_config as sp_create_config
from zoo.board_games.tictactoe.config.tictactoe_muzero_bot_mode_config import main_config as bot_main_config
from zoo.board_games.tictactoe.config.tictactoe_muzero_bot_mode_config import create_config as bot_create_config
from zoo.board_games.tictactoe.config.tictactoe_gumbel_muzero_bot_mode_config import main_config as gumbel_main_config
from zoo.board_games.tictactoe.config.tictactoe_gumbel_muzero_bot_mode_config import create_config as gumbel_create_config
from zoo.board_games.tictactoe.envs.tictactoe_env import TicTacToeEnv
# from ding.utils import set_pkg_seed, ENV_REGISTRY
# from ding.envs import BaseEnv, BaseEnvTimestep
# from ding.envs.env_manager import create_env_manager

# --- PettingZoo Import ---
from pettingzoo.classic import connect_four_v3

CONFIG_TYPE = "gumbel"

def train():
    # set main and create config
    if CONFIG_TYPE == "sp":
        main_config, create_config = sp_main_config, sp_create_config
    elif CONFIG_TYPE == "bot":
        main_config, create_config = bot_main_config, bot_create_config
    elif CONFIG_TYPE == "gumbel":
        main_config, create_config = gumbel_main_config, gumbel_create_config
    else:
        print("invalid config type")

    # Detect GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    main_config.policy.cuda = True
    print(f"Using device: {device}")

    # launch train script
    train_muzero([EasyDict(main_config), EasyDict(create_config)], seed=0)

if __name__ == "__main__":
    train()