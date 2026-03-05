## comparing different strategies
- deep cfr, nfsp: sota for two player, zero sum imperfect information games. weak generalization to multiplayer
- mappo: sota for collaborative, continuous action games
- is-mcts: collects information sets and then rolls out mcts - attempts to adapt mcts to hidden state games
- dqn: bad for non-stationary environments (bad for multiplayer)