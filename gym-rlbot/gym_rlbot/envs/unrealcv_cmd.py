import numpy as np
import cv2
import time
import unrealcv
import io
import PIL.Image
from msvcrt import getch
import argparse
import os
import StringIO

import math
import random

img_width = 80
img_height = 60

class UnrealCv:
    def __init__(self, port = 9000, cam_id = 0,
                 ip = '127.0.0.1'):
        global client
        client = unrealcv.Client((ip, port))
        self.cam_id = cam_id
        self.rotation = (0, 0, 0)
        self.position = [0, 0, 0]
        self.ip = ip
        self.direction = ' '
        self.init_unrealcv()

    def init_unrealcv(self):
        client.connect()
        self.check_connection()
        #client.request('vset /viewmode depth')
        print ('Connected to UE4')
        client.request('vrun setres 640x480')

    def check_connection(self):
        while (client.isconnected() is False):
            print ('UnrealCV server is not running. Please try again')
            client.connect()
    #
    def read_depth(self):
        cmd = 'vget /camera/{cam_id}/depth'
        res = client.request(cmd.format(cam_id=self.cam_id))
        img = PIL.Image.open(StringIO.StringIO(res))
        # img = cv2.resize(np.asarray(img), (img_width, img_height))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # img = img.reshape(1, 1, img_width, img_height)
        # img = np.expand_dims(img,axis=0)
        return img

    def read_depth_npy(self):
        cmd = 'vget /camera/{cam_id}/depth npy'
        res = client.request(cmd.format(cam_id=self.cam_id))
        depth = np.load(io.BytesIO(res))
        # depth = self.convert2planedepth(depth)
        depth[depth>100.0] = 0
        #self.show_img(depth,'depth')
        return np.expand_dims(depth,axis=-1)

    def save_depth(self):
        cmd = 'vget /camera/{cam_id}/depth'
        img_dirs = client.request(cmd.format(cam_id=self.cam_id))
        image = cv2.imread(img_dirs)
        return image

    def convert2planedepth(self,PointDepth, f=320):
        H = PointDepth.shape[0]
        W = PointDepth.shape[1]
        i_c = np.float(H) / 2 - 1
        j_c = np.float(W) / 2 - 1
        columns, rows = np.meshgrid(np.linspace(0, W - 1, num=W), np.linspace(0, H - 1, num=H))
        DistanceFromCenter = ((rows - i_c) ** 2 + (columns - j_c) ** 2) ** (0.5)
        PlaneDepth = PointDepth / (1 + (DistanceFromCenter / f) ** 2) ** (0.5)
        return PlaneDepth

    def get_pose(self):
        cmd = 'vget /camera/{cam_id}/location'
        pos = client.request(cmd.format(cam_id=self.cam_id))
        x,y,z = pos.split()
        self.position = [float(x),float(y),float(z)]
        cmd = 'vget /camera/{cam_id}/rotation'
        ori = client.request(cmd.format(cam_id=self.cam_id))
        pitch,yaw,roll = ori.split()
        self.rotation = (float(roll), float(yaw), float(pitch))
        return [float(x), float(y), float(z), float(yaw)]

    def set_position(self, x, y, z):
        cmd = 'vset /camera/{cam_id}/location {x} {y} {z}'
        client.request(cmd.format(cam_id=self.cam_id, x=x, y=y, z=z))
        self.position = [float(x),float(y),float(z)]

    def set_rotation(self, roll, yaw, pitch):
        cmd = 'vset /camera/{cam_id}/rotation {pitch} {yaw} {roll}'
        client.request(cmd.format(cam_id=self.cam_id, pitch=pitch, yaw=yaw, roll=roll))
        self.rotation = (float(roll), float(yaw), float(pitch))

    def move(self, velocity, angle):

        # continuous motion
        # if (velocity == 1):
        # 	self.keyboard('w')
        # if (angle == 1):
        # 	self.keyboard('a')
        # if (angle == -1):
        # 	self.keyboard('d')
        # if (velocity == -1):
        # 	self.keyboard('s')

        # step by step
        yaw_exp = (self.rotation[1] + angle) % 360
        delt_x = velocity * math.cos(yaw_exp / 180.0 * math.pi)
        delt_y = velocity * math.sin(yaw_exp / 180.0 * math.pi)
        x_exp = self.position[0] + delt_x
        y_exp = self.position[1] + delt_y
        z_exp = self.position[2]

        if angle != 0 :
            self.set_rotation(0, yaw_exp, 0)

        self.set_position(x_exp, y_exp, z_exp)

        #self.notify_ready_for_next_move()
        #time.sleep(0.1)

        return velocity * math.cos(4. * angle / 180.0 * math.pi)

    def keyboard(self,key, duration = 0.1):# Up Down Left Right
        cmd = 'vset /action/keyboard {key} {duration}'
        return client.request(cmd.format(key = key,duration = duration))

    def wait_until_ready(self):
        # wait for UE to be ready
        ready = self.get_ready()
        while (not ready):
           ready = self.get_ready()
        self.reset_ready()

    def wait_until_receive_action(self):
        # get the current steering command
        self.ask_for_action()
        action = self.get_action()
        while (action < 0):
            action = self.get_action()
        self.reset_action()

        return action

    # list of high level TCP commands
    def get_ready(self):
        return client.is_ready()

    def reset_ready(self):
        client.reset_ready()

    def ask_for_action(self):
        self.keyboard('n')

    def get_action(self):
        return client.get_action()

    def reset_action(self):
        client.reset_action()

    def set_ready(self):
        self.keyboard('b')

    def notify_connection(self):
        self.keyboard('c')

    def notify_disconnection(self):
        self.keyboard('l')

    def declare_tick_by_tick(self):
        self.keyboard('t')

    def declare_step_by_step(self):
        self.keyboard('p')

    def reset_env(self):
        self.keyboard('r')

    def get_collision(self):
        return client.is_collided()

    def reset_collision(self):
        client.reset_collision()

    def get_arrival(self):
        return client.is_arrived()

    def reset_arrival(self):
        client.reset_arrival()

    def turn_left(self):
        self.keyboard('a')
        self.direction = 'left'

    def turn_right(self):
        self.keyboard('d')
        self.direction = 'right'

    def forward(self):
        if self.direction == 'left':
            self.keyboard('d')
            self.direction = ' '
        elif self.direction == 'right':
            self.keyboard('a')
            self.direction = ' '
