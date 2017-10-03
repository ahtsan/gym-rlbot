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

img_size = 100

class UnrealCv:
    def __init__(self, port = 9000, cam_id = 0,
                 ip = '127.0.0.1'):
        global client
        client = unrealcv.Client((ip, port))
        self.cam_id = cam_id
        self.ip = ip
        self.direction = ' '
        self.init_unrealcv()

    def init_unrealcv(self):
        client.connect()
        self.check_connection()
        #client.request('vset /viewmode depth')
        time.sleep(1)
        print ('Connected to UE4')

    def check_connection(self):
        while (client.isconnected() is False):
            print ('UnrealCV server is not running. Please try again')
            client.connect()

    def read_depth(self):
        cmd = 'vget /camera/{cam_id}/depth png'
        res = client.request(cmd.format(cam_id=self.cam_id))
        img = PIL.Image.open(StringIO.StringIO(res))
        img = cv2.resize(np.asarray(img), (img_size, img_size))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = img.reshape(img_size, img_size, 1)
        img = np.expand_dims(img,axis=0)
        return img

    def save_depth(self):
        cmd = 'vget /camera/{cam_id}/depth'
        img_dirs = client.request(cmd.format(cam_id=self.cam_id))
        image = cv2.imread(img_dirs)
        return image

    def get_pose(self):
        cmd = 'vget /camera/{cam_id}/location'
        pos = client.request(cmd.format(cam_id=self.cam_id))
        x,y,z = pos.split()
        cmd = 'vget /camera/{cam_id}/rotation'
        ori = client.request(cmd.format(cam_id=self.cam_id))
        pitch,yaw,roll = ori.split()
        return [float(x), float(y), float(z), float(yaw)]

    def move(self,cam_id, velocity, angle):

        if (velocity == 1):
        	self.keyboard('w')
        if (angle == 1):
        	self.keyboard('a')
        if (angle == -1):
        	self.keyboard('d')
        if (velocity == -1):
        	self.keyboard('s')

        #self.notify_ready_for_next_move()
        #time.sleep(0.1)

        return False

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

    def declare_tick_by_tick(self):
        self.keyboard('t')

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
