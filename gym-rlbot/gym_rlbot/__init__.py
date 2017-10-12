from gym.envs.registration import register

register(
    id='rlbot-v0',
    entry_point='gym_rlbot.envs:RLBOT_base',
    kwargs = {'setting_file' : 'goal_freemove.json',
              'test' : True,
              'action_type' : 'discrete',
              'observation_type' : 'depth',
              'reward_type': 'no'
              }
)

register(
    id='rlbot-v1',
    entry_point='gym_rlbot.envs:RLBOT_base_new',
    kwargs = {'setting_file' : 'goal_freemove.json',
              'test' : True,
              'action_type' : 'discrete',
              'observation_type' : 'depth',
              'reward_type': 'no'
              }
)
