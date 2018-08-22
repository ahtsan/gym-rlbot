import gym
import cv2
from unrealcv_cmd import UnrealCv
import numpy as np
import time
import random
import math
from gym import spaces
import os
from operator import itemgetter

INPUT_WIDTH = 80
INPUT_HEIGHT = 60

class Rlbot_env_free(gym.Env):
   def __init__(self,
                setting_file,
                test,               # if True will use the test_xy as start point
                timeDependent,
                action_type,  # 'discrete', 'continuous'
                observation_type, # 'color', 'depth', 'rgbd'
                reward_type # distance, bbox, bbox_distance,
                ):

     self.show = False
     self.collision = False
     self.timeDependent = timeDependent
     self.target = None

     print ("Reading setting file: ",setting_file)
     self.action_type = action_type
     self.observation_type = observation_type
     self.reward_type = reward_type
     assert self.action_type == 'discrete' or self.action_type == 'continuous'
     assert self.observation_type == 'depth'
     setting = self.load_env_setting(setting_file)

     # connect UnrealCV
     self.unrealcv = UnrealCv(9000, self.cam_id, self.env_ip, self.message_handler)

     print ("Camera_id: ",self.cam_id);
     print ("Environment ip: ",self.env_ip)
     print ("Observation type: ", observation_type)
     print ("action_type: ",action_type)
     print ("Reward type: ",reward_type)

     self.unrealcv.notify_connection()

     self.count_steps = 0

     # action
     if self.action_type == 'discrete':
         self.action_space = spaces.Discrete(len(self.discrete_angular) * len(self.discrete_linear))
     elif self.action_type == 'continuous':
         self.action_space = spaces.Box(low = np.array(self.continous_actions['low']),high = np.array(self.continous_actions['high']))

     # get start position
     self.startPoint = self.unrealcv.get_pose()
     self.unrealcv.get_target_point()

     # while(self.target == None):
     #     time.sleep(0.5)

     # for reset point generation
     self.trajectory = []
     self.start_id = 0

   def _step(self, actions):
        info = dict(
            Collision=False,
            Arrival=False,
            Done = False,
            Maxstep=False,
            Reward=0.0,
            Action = actions,
            Pose = [],
            Trajectory = self.trajectory,
            Steps = self.count_steps,
            Target = [],
            Distance = 0.0,
            Direction = 0.0,
            Color = None,
            Depth = None,
        )

        action_angular = actions[0]
        action_linear = actions[1]
        # return state immediately
        if action_angular == -1:
            if self.observation_type == 'depth':
                state = info['Depth'] = self.unrealcv.read_depth_npy()

            resized_state = cv2.resize(state, (INPUT_WIDTH, INPUT_HEIGHT))
            resized_state = np.expand_dims(resized_state, axis=2)
            return resized_state, 0, False, info
        # disconnect UE
        elif action_angular == -2:
            if self.observation_type == 'depth':
                state = info['Depth'] = self.unrealcv.read_depth_npy()

            resized_state = cv2.resize(state, (INPUT_WIDTH, INPUT_HEIGHT))
            resized_state = np.expand_dims(resized_state, axis=2)
            self.unrealcv.notify_disconnection()
            return resized_state, 0, False, info

        if self.action_type == 'discrete':
            (angle, velocity) = self.discrete_angular[action_angular], self.discrete_linear[action_linear]
        else:
            (angle, velocity) = None

        if angle == 0 and velocity == 0:
            angle = 10
            
        info['Done'] = False

        # take action
        distance_travelled = self.unrealcv.move(velocity, angle, False)

        info['Pose'] = self.unrealcv.get_pose()
        # info['Distance'] = self.get_distance(info['Pose'][:3],self.targets_pos)
        if self.reward_type == 'distance':
            info['Reward'] = self.reward_distance(info['Pose'], angle)
        elif self.reward_type == 'action':
            info['Reward'] = self.reward_action(velocity, angle)
        else:
            info['Reward'] = distance_travelled / 1000

        # info['Direction'] = self.get_direction (info['Pose'],self.targets_pos)
        info['Collision'] = self.isCollided()

        # info['Arrival'] = self.unrealcv.get_arrival()
        # self.unrealcv.reset_arrival()

        if info['Collision']:
            info['Reward'] = -1
            info['Done'] = True

        # print ("R: ", str(round(info['Reward'], 3)))
        # print ("R: ", str(round(info['Reward'], 3)), " D: ", str(round(info['Distance'], 3)), " A: ", str(round(info['Direction'], 3)))
        self.count_steps += 1

        # limit the max steps of every episode
        if self.count_steps > self.max_steps:
           info['Done'] = True
           info['Maxstep'] = True
           print '@@@ Reach Max Steps'

        # save the trajectory
        self.trajectory.append(info['Pose'])
        info['Trajectory'] = self.trajectory

        # update observation
        if self.observation_type == 'depth':
            state = info['Depth'] = self.unrealcv.read_depth_npy()

        if self.show:
            state = cv2.resize(state, (432, 243))
            cv2.imshow('state', state / np.max(state))
            cv2.waitKey(1)

        resized_state = cv2.resize(state, (INPUT_WIDTH, INPUT_HEIGHT))
        resized_state = np.expand_dims(resized_state, axis=2)
        # return np.array([state, np.array([info['Distance']])]), info['Reward'], info['Done'], info
        return resized_state, info['Reward'], info['Done'], info

   def _reset(self, ):
       self.unrealcv.reset_env()

    #    ready = self.unrealcv.get_ready()
    #    while (not ready):
    #        ready = self.unrealcv.get_ready()
    #    self.unrealcv.reset_ready()

       if self.test == 1:
           current_pose = self.reset_from_testpoint()
       else:
           # current_pose = self.unrealcv.get_pose()
           current_pose = self.reset_from_startPoint()

       if self.observation_type == 'depth':
           state = self.unrealcv.read_depth_npy()

       self.trajectory = []
       self.trajectory.append(current_pose)
       self.count_steps = 0

       #angle = self.get_direction (current_pose,self.targets_pos)
       resized_state = cv2.resize(state, (INPUT_WIDTH, INPUT_HEIGHT))
       resized_state = np.expand_dims(resized_state, axis=2)

       return resized_state

   def _close(self):
       pass

   def message_handler(self, message):
       if (message == 'collision'):
           self.collision = True
       if (message.startswith('nextTarget')):
           tmp = message.split(':')[1].split(',')
           self.target = [float(tmp[0]), float(tmp[1])]
           self.isTargetChanged = True

   def isCollided(self):
       collided = self.collision
       self.collision = False
       return collided

   # functions for starting point module
   def reset_from_testpoint(self):
       x,y = self.testpoints[self.start_id]
       z = self.height
       # noise
       x += random.uniform(-500, 2000)
       y += random.uniform(-80, 80)
       yaw = random.uniform(-45, 45)

       self.unrealcv.set_position(x, y, z)
       self.unrealcv.set_rotation(0, yaw, 0)
       return [x,y,z,yaw]

   def reset_from_startPoint(self):
      x, y, z, yaw = self.startPoint
      # noise
      # x += random.uniform(-500, 2000)
      # y += random.uniform(-80, 80)
      # yaw = random.uniform(-45, 45)

      self.unrealcv.set_position(x, y, z)
      self.unrealcv.set_rotation(0, yaw, 0)
      return [x,y,z,yaw]

   def get_distance(self, current, target):
       dis = abs(np.array(target)[:2] - np.array(current)[:2])# only x and y
       distance = math.sqrt(sum(dis * dis))
       return distance

   # def get_direction(self,current_pose,target_pose):
   #     y_delt = target_pose[1] - current_pose[1]
   #     x_delt = target_pose[0] - current_pose[0]
   #     if x_delt == 0:
   #         x_delt = 0.00001
   #
   #     target_angle = np.arctan(y_delt / x_delt) / 3.1415926 * 180
   #
   #     if (y_delt > 0 and x_delt < 0):
   #         target_angle += 180
   #     if (y_delt > 0 and x_delt > 0):
   #         target_angle = 360 - target_angle
   #     if (y_delt < 0 and x_delt > 0):
   #         target_angle += 360
   #
   #     difference = target_angle - current_pose[-1]
   #
   #     if (difference > 180):
   #         difference -= 360
   #
   #     #The angle difference: (-ve) means should rotate to left
   #     #(+ve) means should rotate to right
   #     #range is (-180,180]
   #     return difference

   def reward_distance(self, current_pose, angle):
       if self.isTargetChanged:
          self.dis2target_now= self.get_distance(current_pose, self.target)
          self.isTargetChanged = False
          reward = 0
       elif angle != 0:
           reward = 0
       else:
           distance = self.get_distance(current_pose, self.target)
           reward = (self.dis2target_now - distance) / 1000
           self.dis2target_now = distance

       return reward

   def reward_action(self, velocity, angle):
       if angle == 0 and velocity != 0:
           return 0.02
       elif angle != 0 and velocity != 0:
           return 0.01
       elif angle != 0 and velocity == 0:
           return 0
       else:
           return -0.01

   def load_env_setting(self,filename):
       f = open(self.get_settingpath(filename))
       type = os.path.splitext(filename)[1]
       if type == '.json':
           import json
           setting = json.load(f)
       elif type == '.yaml':
           import yaml
           setting = yaml.load(f)
       else:
           print 'unknown type'


       print setting
       self.cam_id = setting['cam_id']
       self.env_ip = setting['env_ip']
       self.max_steps = setting['max_steps']
       self.height = setting['height']
       self.testpoints = setting['test_xy']
       self.discrete_actions = setting['discrete_actions']
       self.discrete_angular = setting['discrete_angular']
       self.discrete_linear = setting['discrete_linear']
       self.continous_actions = setting['continous_actions']
       self.test = setting['test']

       return setting

   def get_settingpath(self, filename):
       import gym_rlbot
       gympath = os.path.dirname(gym_rlbot.__file__)
       return os.path.join(gympath, 'envs/setting', filename)
