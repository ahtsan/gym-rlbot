import numpy as np
import time
import unrealcv
import cv2
from msvcrt import getch
import argparse
import os
from unrealcv_cmd import UnrealCv

img_size = 100
img_size_display = 500

if __name__ == "__main__":

    UE = UnrealCv()

    while (True):
        image = UE.read_depth_npy_py35()
        cv2.imshow('image', image / np.max(image))
        cv2.waitKey(1)
