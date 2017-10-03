import numpy as np
import cv2
import time
import unrealcv
import io
import PIL.Image
from msvcrt import getch
import argparse
import os
from keras.models import model_from_json
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from unrealcv_cmd import UnrealCV

img_size = 100
img_size_display = 500
tick_by_tick = False
show_image = True

if __name__ == "__main__":
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    print("Loaded model from disk")
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model.h5")

    UE = UnrealCV()

    UE.notify_connection()
    if tick_by_tick:
        UE.declare_tick_by_tick()

    #get ready
    image = UE.read_depth()
    image = image.astype(float)
    score = loaded_model.predict(image/255)

    UE.reset_env()

    episode = 0
    arrival_epi = 0
    collided_epi = 0

    while(True):
        if tick_by_tick:
            UE.wait_until_ready()

        image = UE.read_depth()
        image = image.astype(float)
        score = loaded_model.predict(image)
        action_index = np.argmax(score[0])

        if show_image:
            resized_image = cv2.resize(image.reshape(img_size, img_size), (img_size_display, img_size_display))
            resized_image = resized_image.astype(np.uint8)
            cv2.imshow('image', resized_image)
            cv2.waitKey(1)

        if action_index == 0:
            UE.turn_left()
        if action_index == 1:
            UE.forward()
        if action_index == 2:
            UE.turn_right()

        collision = UE.get_collision()
        UE.reset_collision()

        arrival = UE.get_arrival()
        UE.reset_arrival()

        if collision:
            UE.reset_env()
            episode += 1
            collided_epi += 1
            print ('Success: {0}/{2}, Failed: {1}/{2}, performance: {3}%'.format(arrival_epi, collided_epi, episode, float(arrival_epi)*100/episode))
        elif arrival:
            UE.reset_env()
            episode += 1
            arrival_epi += 1
            print ('Success: {0}/{2}, Failed: {1}/{2}, performance: {3}%'.format(arrival_epi, collided_epi, episode, float(arrival_epi)*100/episode))
        else:
            if tick_by_tick:
                UE.set_ready()
