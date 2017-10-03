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

img_size = 100
img_size_display = 500

if __name__ == "__main__":
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    print("Loaded model from disk")
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model.h5")

    images = np.load('rlbot_training_dataset.npy')
    print ('Training dataset sample size: ' + str(images.shape[0]))
    print ('Sample dimension: ' + str(images.shape[1]))

    index = np.arange(0, len(images))

    for i in index:
        print ('Testing on image' + str(i))
        # process the image
        image = images[i].reshape(img_size, img_size, 1)
        new_image = np.expand_dims(image, axis=0)

        score = loaded_model.predict(new_image/255)
        action_index = np.argmax(score[0])

        resized_image = cv2.resize(new_image.reshape(img_size, img_size), (img_size_display, img_size_display))
        resized_image = resized_image.astype(np.uint8)
        cv2.imshow('image', resized_image)
        cv2.waitKey(0)

        action = None
        if action_index == 0:
            action = 'left'
        if action_index == 1:
            action = 'forward'
        if action_index == 2:
            action = 'right'

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(resized_image,action,(30,450), font, 3,(255,255,255),2,cv2.LINE_AA)
        cv2.imshow('image', resized_image)
        cv2.waitKey(0)
