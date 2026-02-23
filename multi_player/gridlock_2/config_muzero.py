from easydict import EasyDict


# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
env_id = 'gridlock2'
action_space_size = 9
collector_env_num = 16
n_episode = 16
evaluator_env_num = 10
num_simulations = 150
update_per_collect = 200
batch_size = 512
max_env_step = int(1e7)
reanalyze_ratio = 0.
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

gridlock2_gumbel_muzero_config = dict(
    exp_name=f'data_varied_muzero/gridlock2_muzero_ns{num_simulations}_upc{update_per_collect}_rer{reanalyze_ratio}_bs{batch_size}_seed0',
    env=dict(
        env_id=env_id,
        num_players=2,
        battle_mode='self_play_mode',
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
        # prob_random_agent=0.07,
        # prob_expert_agent=0.13,
    ),
    policy=dict(
        model=dict(
            observation_shape=(3,3,3),
            action_space_size=action_space_size,
            image_channel=3,
            num_res_blocks=3,
            num_channels=64,
            reward_head_hidden_channels=[32],
            value_head_hidden_channels=[32],
            policy_head_hidden_channels=[32],
            # NOTE: whether to use the self_supervised_learning_loss. default is False
            # self_supervised_learning_loss=True,
        ),
        cuda=True,
        env_type='board_games',
        action_type='varied_action_space',
        game_segment_length=18,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        piecewise_decay_lr_scheduler=False,
        learning_rate=0.001,
        grad_clip_value=0.5,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        max_num_considered_actions=5,
        td_steps=10,
        num_unroll_steps=5,
        discount_factor=1,
        # manual_temperature_decay=True,
        # threshold_training_steps_for_final_temperature=int(1e5),
        # (float) Weight decay for training policy network.
        # weight_decay=1e-4,
        # ssl_loss_weight=2,  # default is 0
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
gridlock2_gumbel_muzero_config = EasyDict(gridlock2_gumbel_muzero_config)
main_config = gridlock2_gumbel_muzero_config

gridlock2_gumbel_muzero_create_config = dict(
    env=dict(
        type='gridlock2',
        import_names=['env_eval'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='muzero',
        import_names=['lzero.policy.muzero'],
    ),
)
gridlock2_gumbel_muzero_create_config = EasyDict(gridlock2_gumbel_muzero_create_config)
create_config = gridlock2_gumbel_muzero_create_config

def get_gridlock2_config():
    return main_config, create_config
