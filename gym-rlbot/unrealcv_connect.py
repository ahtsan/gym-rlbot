import numpy as np
import time
import unrealcv
import cv2
import StringIO, PIL.Image
from msvcrt import getch
import argparse
import os
import argparse
from gym_rlbot.envs.unrealcv_cmd import UnrealCv

show_FPS = True
UE = None
def message_handler(message):
    print ('received: ' + message)
    if (message == 'collision'):
        UE.reset_env()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='show image', action='store_true')
    parser.add_argument('--fps', help='show image', action='store_true')
    args = parser.parse_args()

    UE = UnrealCv(message_handler=message_handler)
    UE.notify_connection()
    start_time = time.time()
    count = 1
    num = 0

    if args.image:
        while(True):
            # get the depth image
            image = UE.read_depth_npy()
            image = cv2.resize(image.reshape(81, 144), (640, 480))
            # image = image / np.max(image)
            # print (np.min(image))
            # print (np.max(image))
            # image = cv2.normalize(image, image, 0, 1, cv2.NORM_MINMAX)
            print ("min: " + str(np.min(image)))
            print ("max: " + str(np.max(image)))
            cv2.imshow('image', image / np.max(image))
            cv2.waitKey(1)
    elif args.fps:
        while(True):
            # get the depth image
            image = UE.read_depth_npy()
            num += 1
            elapsed_time = time.time() - start_time
            if (elapsed_time > count):
                count += 1
                print ("FPS: " + str(num))
                num = 0
    else:
        input()
