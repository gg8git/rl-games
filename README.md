# rl-games

trying to use policy gradient rl methods to learn to play games

## games to learn

single-player, turn based games
- gridlock

multi-player, turn based, symmetric games
- gong zhu
- coup
- catan

## notes about on policy RL

bunch of relevant fields / subfields
1. single agent control (TRPO, PPO) - how to get single agent to train & converge reliably
2. strategic solvers & marl (mcts, cfr, psro) - how to deal with moving target
3. meta RL (maml, RL^2) - how to learn more efficiently

within marl:
1. search (mcts, alphazero, muzero, etc)
2. policy gradient (ppo, mappo, a2c, etc)
3. value (q-learning, dqn, etc)
4. regret (cfr, deep cfr, etc)
5. population management (psro, nfsp, neuRD, etc)

adaptation:
1. implicit opponent modeling (meta-RL, RL^2, opponent transformer, LOSI)
2. explicit opponent modeling (DRON, bayesian opponent modeling, OMIS)
https://arxiv.org/pdf/2502.04686

next steps: 
1. learn more about deep cfr and neuRD
2. research adaptation+quick learning