"""Train AlphaZero on Connect4."""

from absl import app
from absl import flags
from open_spiel.python.algorithms.alpha_zero import alpha_zero
from open_spiel.python.algorithms.alpha_zero import utils
from open_spiel.python.utils import spawn
from datetime import datetime

flags.DEFINE_enum("nn_api", "linen", ["linen", "nnx"], "Flax API to use.")
flags.DEFINE_string("game", "connect_four", "Name of the game.")
flags.DEFINE_float("uct_c", 2.0, "UCT's exploration constant.")
flags.DEFINE_integer("max_simulations", 400, "Simulations per MCTS search.")
flags.DEFINE_integer("train_batch_size", 256, "Batch size for learning.")
flags.DEFINE_integer("replay_buffer_size", 2**15, "Replay buffer size.")
flags.DEFINE_integer("replay_buffer_reuse", 3, "Learning frequency per state.")
flags.DEFINE_float("learning_rate", 1e-3, "Learning rate.")
flags.DEFINE_float("weight_decay", 1e-4, "L2 regularization strength.")
flags.DEFINE_bool("decouple_weight_decay", False, "Decouple weights.")
flags.DEFINE_float("policy_epsilon", 0.25, "Noise epsilon.")
flags.DEFINE_float("policy_alpha", 1.0, "Dirichlet noise alpha.")
flags.DEFINE_float("temperature", 1.0, "Temperature for move selection.")
flags.DEFINE_integer("temperature_drop", 10, "Drop temp to 0 after N moves.")
flags.DEFINE_enum("nn_model", "resnet", utils.api_selector(utils.AVIALABLE_APIS[0]).Model.valid_model_types, "Model type.")
flags.DEFINE_integer("nn_width", 128, "Network width (filters).")
flags.DEFINE_integer("nn_depth", 6, "Network depth (residual blocks).")
flags.DEFINE_string("path", f"/workspace/connect_four/runs/az_c4_new", "Where to save checkpoints.")
flags.DEFINE_integer("checkpoint_freq", 50, "Save checkpoint every N steps.")
flags.DEFINE_integer("actors", 4, "Number of actors for self-play.")
flags.DEFINE_integer("evaluators", 2, "Number of evaluators.")
flags.DEFINE_integer("evaluation_window", 50, "Games to average results over.")
flags.DEFINE_integer("eval_levels", 5, "MCTS evaluation levels.")
flags.DEFINE_integer("max_steps", 1000, "Learn steps before exiting.")
flags.DEFINE_bool("quiet", True, "Don't show played moves.")
flags.DEFINE_bool("verbose", False, "Show MCTS stats.")

FLAGS = flags.FLAGS

def main(unused_argv):
  config = alpha_zero.Config(
      game=FLAGS.game,
      path=FLAGS.path,
      learning_rate=FLAGS.learning_rate,
      weight_decay=FLAGS.weight_decay,
      decouple_weight_decay=FLAGS.decouple_weight_decay,
      train_batch_size=FLAGS.train_batch_size,
      replay_buffer_size=FLAGS.replay_buffer_size,
      replay_buffer_reuse=FLAGS.replay_buffer_reuse,
      max_steps=FLAGS.max_steps,
      checkpoint_freq=FLAGS.checkpoint_freq,
      actors=FLAGS.actors,
      evaluators=FLAGS.evaluators,
      uct_c=FLAGS.uct_c,
      max_simulations=FLAGS.max_simulations,
      policy_alpha=FLAGS.policy_alpha,
      policy_epsilon=FLAGS.policy_epsilon,
      temperature=FLAGS.temperature,
      temperature_drop=FLAGS.temperature_drop,
      evaluation_window=FLAGS.evaluation_window,
      eval_levels=FLAGS.eval_levels,
      nn_model=FLAGS.nn_model,
      nn_width=FLAGS.nn_width,
      nn_depth=FLAGS.nn_depth,
      observation_shape=None,
      output_size=None,
      quiet=FLAGS.quiet,
      verbose=FLAGS.verbose,
      nn_api_version=FLAGS.nn_api,
  )
  alpha_zero.alpha_zero(config)

if __name__ == "__main__":
  with spawn.main_handler():
    app.run(main)