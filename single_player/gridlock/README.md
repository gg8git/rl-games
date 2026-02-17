# gridlock

## rules

gridlock:
- random draw of 40 cards (1 through 10, 4x of each)
- draw cards one at a time and play in 3x3 grid
- cards cannot go over other cards, cards must be strictly greater than the cards directly below and left of them, cards must be strictly less than the cards directly above and right of them
- game is over once 9 cards have been played or the next card cannot be placed anywhere in the grid
- score is the sum of each completed column, row, and diagonal

## homemade training loop

`homemade_ppo_sft.py` and `homemade_demo.py` are a homemade rollout of sft+ppo on this task

## cleanrl training loop

`cleanrl_ppo_sft.py`, `cleanrl_gym_env.py`, and `env_demo.py` are a cleanrl library supported rollout of sft+ppo on this task

## cleanrl training loop

`sb3_ppo_sft.py`, `sb3_gym_env.py`, and `env_demo.py` are a cleanrl library supported rollout of sft+ppo on this task