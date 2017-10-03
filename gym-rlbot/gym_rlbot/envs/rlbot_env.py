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

'''
It is a general env for searching target object.

State : raw color image and depth (640x480)
Action:  (linear velocity ,angle velocity)
Done : Collision or get target place or False trigger three times.
Task: Learn to avoid obstacle and search for a target object in a room,
      you can select the target name according to the Recommend object list as below

Recommend object list in RealisticRendering
 'SM_CoffeeTable_14', 'Couch_13','SM_Couch_1seat_5','Statue_48','SM_TV_5', 'SM_DeskLamp_5'
 'SM_Plant_7', 'SM_Plant_8', 'SM_Door_37', 'SM_Door_39', 'SM_Door_41'

Recommend object list in Arch1
'BP_door_001_C_0','BP_door_002_C_0'
'''

class RLBOT_base(gym.Env):
   def __init__(self,
                setting_file,
                test = False,               # if True will use the test_xy as start point
                action_type = 'discrete',  # 'discrete', 'continuous'
                observation_type = 'depth', # 'color', 'depth', 'rgbd'
                reward_type = 'distance' # distance, bbox, bbox_distance,
                ):

     print ("Reading setting file: ",setting_file)
     self.test = test
     self.action_type = action_type
     self.observation_type = observation_type
     self.reward_type = reward_type
     assert self.action_type == 'discrete' or self.action_type == 'continuous'
     assert self.observation_type == 'depth'
     setting = self.load_env_setting(setting_file)

     # connect UnrealCV
     self.unrealcv = UnrealCv(9000, self.cam_id, self.env_ip)

     print ("Camera_id: ",self.cam_id);
     print ("Environment ip: ",self.env_ip)
     print ("Observation type: ", observation_type)
     print ("action_type: ",action_type)
     print ("Reward type: ",reward_type)

     self.unrealcv.notify_connection()

     self.count_steps = 0

     self.targets_pos = self.target_points[0]
     print ("Target points: ",self.target_points)
     print ("The target pose: ",self.targets_pos)

     # action
     if self.action_type == 'discrete':
         self.action_space = spaces.Discrete(len(self.discrete_actions))
     elif self.action_type == 'continuous':
         self.action_space = spaces.Box(low = np.array(self.continous_actions['low']),high = np.array(self.continous_actions['high']))

    # observation
    #  if self.observation_type == 'depth':
    #      state = self.unrealcv.read_depth()
        #  self.imageList = imageList.imageList(1, state)     # initialize input list ( 3 depth images )
        #  self.imageList.setup(self.observation_space)

     # set start position
     current_pose = self.unrealcv.get_pose()

     self.distance_last = self.get_distance(current_pose, self.targets_pos)

     # for reset point generation
     self.trajectory = []
     self.start_id = 0
     self.yaw_id = 0

     self.collision = False

   def _step(self, action):
        info = dict(
            Collision=False,
            Done = False,
            Maxstep=False,
            Reward=0.0,
            Action = action,
            Pose = [],
            Trajectory = self.trajectory,
            Steps = self.count_steps,
            Target = [],
            Distance = 0.0,
            Direction = 0.0,
            Color = None,
            Depth = None,
        )

        self.count_steps += 1

        if self.action_type == 'discrete':
            (velocity, angle) = self.discrete_actions[action]
        else:
            (velocity, angle) = action

        info['Done'] = False

        # take action
        self.unrealcv.move(self.cam_id, velocity, angle)

        # ready = self.unrealcv.get_ready()
        # while (not ready):
        #     ready = self.unrealcv.get_ready()
        # self.unrealcv.reset_ready()

        # update observation
        if self.observation_type == 'depth':
            state = info['Depth'] = self.unrealcv.read_depth()

        # get reward
        info['Pose'] = self.unrealcv.get_pose()
        info['Distance'] = self.get_distance(info['Pose'][:3],self.targets_pos)
        if self.reward_type=='distance':
            info['Reward'] = self.reward_distance(info['Distance'])
        else:
            info['Reward'] = 0

        # info['Direction'] = self.get_direction (info['Pose'],self.targets_pos)
        info['Collision'] = self.unrealcv.get_collision()
        self.unrealcv.reset_collision()

        arrival = self.unrealcv.get_arrival()
        self.unrealcv.reset_arrival()

        if info['Collision']:
            info['Reward'] = -1
            info['Done'] = True
        elif arrival:
            info['Reward'] = 1
            info['Done'] = True

        #print ("R: ", str(round(info['Reward'], 3)), " D: ", str(round(info['Distance'], 3)), " A: ", str(round(info['Direction'], 3)))

        # limit the max steps of every episode
        if self.count_steps > self.max_steps:
           info['Done'] = True
           info['Maxstep'] = True
           print '@@@ Reach Max Steps'

        # save the trajectory
        self.trajectory.append(info['Pose'])
        info['Trajectory'] = self.trajectory

        resized_image = cv2.resize(state.reshape(100, 100), (500, 500))
        resized_image = resized_image.astype(np.uint8)
        cv2.imshow('image', resized_image)
        cv2.waitKey(1)

        # return np.array([state, np.array([info['Distance']])]), info['Reward'], info['Done'], info
        return state, info['Reward'], info['Done'], info

   def _reset(self, ):
       self.unrealcv.reset_env()

    #    ready = self.unrealcv.get_ready()
    #    while (not ready):
    #        ready = self.unrealcv.get_ready()
    #    self.unrealcv.reset_ready()

       if self.test:
           current_pose = self.reset_from_testpoint()
       else:
           current_pose = self.unrealcv.get_pose()

       if self.observation_type == 'depth':
           state = self.unrealcv.read_depth()

       self.trajectory = []
       self.trajectory.append(current_pose)
       self.count_steps = 0
       self.distance_last = self.get_distance(current_pose, self.targets_pos)

       #angle = self.get_direction (current_pose,self.targets_pos)

       return state

   def _close(self):
       pass

   # def _get_action_size(self):
   #     return len(self.action)

   # functions for starting point module
   def reset_from_testpoint(self):
       x,y = self.testpoints[self.start_id]
       z = self.height
       #yaw = self.yaw_id * 45
       yaw = 0
       self.unrealcv.set_position(self.cam_id, x, y, z)
       self.unrealcv.set_rotation(self.cam_id, 0, yaw, 0)
       self.yaw_id += 1
       if self.yaw_id >=8:
           self.start_id = (self.start_id + 1) % len(self.testpoints)
           self.yaw_id = 0
       return [x,y,z,yaw]

   def get_distance(self,current,target):
       error = abs(np.array(target)[:2] - np.array(current)[:2])# only x and y
       distance = math.sqrt(sum(error * error))
       #distance = abs(target[0] - current[0])
       return distance

   def get_direction(self,current_pose,target_pose):
       y_delt = target_pose[1] - current_pose[1]
       x_delt = target_pose[0] - current_pose[0]
       if x_delt == 0:
           x_delt = 0.00001

       target_angle = np.arctan(y_delt / x_delt) / 3.1415926 * 180

       if (y_delt > 0 and x_delt < 0):
           target_angle += 180
       if (y_delt > 0 and x_delt > 0):
           target_angle = 360 - target_angle
       if (y_delt < 0 and x_delt > 0):
           target_angle += 360

       difference = target_angle - current_pose[-1]

       if (difference > 180):
           difference -= 360

       #The angle difference: (-ve) means should rotate to left
       #(+ve) means should rotate to right
       #range is (-180,180]
       return difference

   def reward_distance(self,distance_now):

       reward = (self.distance_last - distance_now) / 50.0
       self.distance_last = distance_now

       return reward

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
       self.target_points = setting['target_points']
       self.max_steps = setting['maxsteps']
       self.height = setting['height']
       self.testpoints = setting['test_xy']
       self.discrete_actions = setting['discrete_actions']
       self.continous_actions = setting['continous_actions']

       return setting

   def get_settingpath(self, filename):
       import gym_rlbot
       gympath = os.path.dirname(gym_rlbot.__file__)
       return os.path.join(gympath, 'envs/setting', filename)
