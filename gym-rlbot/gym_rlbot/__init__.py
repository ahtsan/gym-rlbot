from gym.envs.registration import register

register(
    id='rlbot-v1',
    entry_point='gym_rlbot.envs:Rlbot_env_free',
    kwargs = {'setting_file' : 'goal_freemove.json',
              'test' : False,
              'timeDependent': False,
              'action_type' : 'discrete',
              'observation_type' : 'depth',
              'reward_type': 'action'
              }
)

register(
    id='rlbot-v2',
    entry_point='gym_rlbot.envs:Rlbot_env_loco',
    kwargs = {'setting_file' : 'goal_loco.json',
              'test' : False,
              'timeDependent': True,
              'action_type' : 'discrete',
              'observation_type' : 'depth',
              'reward_type': 'action'
              }
)
