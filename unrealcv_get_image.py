import numpy as np
import time
import unrealcv
import cv2
import StringIO, PIL.Image
from msvcrt import getch
import argparse
import os
from unrealcv_cmd import UnrealCV

img_size = 100
img_size_display = 500

if __name__ == "__main__":

    UE = UnrealCV()

    while (True):
        image = UE.read_depth()
        image = image.astype(float)

        resized_image = cv2.resize(image.reshape(img_size, img_size), (img_size_display, img_size_display))
        resized_image = resized_image.astype(np.uint8)
        cv2.imshow('image', resized_image)
        cv2.waitKey(0)
