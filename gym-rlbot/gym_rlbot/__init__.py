from gym.envs.registration import register

register(
    id='rlbot-v0',
    entry_point='gym_rlbot.envs:RLBOT_base',
    kwargs = {'setting_file' : 'goal.json',
              'test' : False,
              'action_type' : 'discrete',
              'observation_type' : 'depth',
              'reward_type': 'goal'
              }
)
