import numpy as np
import time
import unrealcv
import cv2
import StringIO, PIL.Image
from msvcrt import getch
import argparse
import os
from unrealcv_cmd import UnrealCV

count = -1
labels = None
result = None
img_size = 100
img_size_display = 500
show_image = True
tick_by_tick = False

# for lstm
final_data = None
timestamp = 3
input_shape = int(timestamp * img_size * img_size)
remove_shape = int((timestamp - 1) * img_size * img_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lstm", help="plot weekly graph",action="store_true")
    parser.add_argument('number_of_samples',type=int)
    args = parser.parse_args()

    if args.lstm:
        print ('Start collecting data for lstm.')
    else:
        print ('Start collecting data for cnn.')

    number_of_samples = args.number_of_samples

    UE = UnrealCV()

    UE.notify_connection()
    if tick_by_tick:
        UE.declare_tick_by_tick()

    UE.reset_env()

    while(count < number_of_samples):

        if tick_by_tick:
            UE.wait_until_ready()

        # get the depth image
        image = UE.read_depth()
        image = image.astype(float)
        
        # get the current steering command
        action = UE.wait_until_receive_action()

        # check terminating state
        collision = UE.get_collision()
        UE.reset_collision()

        arrival = UE.get_arrival()
        UE.reset_arrival()

        if show_image:
            resized_image = cv2.resize(image.reshape(img_size, img_size), (img_size_display, img_size_display))
            resized_image = resized_image.astype(np.uint8)
            cv2.imshow('image', resized_image)
            cv2.waitKey(1)

        if (args.lstm): # for lstm
            if (not collision and not arrival):
                # save the depth image
                if (count >= 0): # skip the first image
                    if (images is None):
                        images = np.array([image])
                    else:
                        if images.shape[0] == 3:
                            images = np.delete(images, 0, 0)
                        images = np.concatenate((images, np.array([image])), axis = 0)

                    if (result is None):
                        result = image.flatten()
                    else:
                        if result.shape[0] == input_shape:
                            result = result[-remove_shape:]
                        result = np.append(result, image.flatten())

                    if result.shape[0] == input_shape:
                        # save the result
                        if final_data is None:
                            final_data = result
                        else:
                            final_data = np.vstack((final_data, result))
                        # save the label
                        if action == 3:
                            label = [0., 0., 1.]
                        elif action == 2:
                            label = [0., 1., 0.]
                        elif action == 1:
                            label = [1., 0., 0.]

                        if (labels is None):
                            labels = label
                        else:
                            labels = np.vstack((labels, label))

                count += 1
                print ('count: ',  count)
                # move on
                if tick_by_tick:
                    UE.set_ready()
            else:
                result = None
                UE.reset_env()

        else:
            if (not collision and not arrival):
                if (count >= 0): # skip the first image
                    if (result is None):
                        result = image.flatten()
                    else:
                        result = np.vstack((result, image.flatten()))

                    # save the label
                    if action == 3:
                        label = [0., 0., 1.]
                    elif action == 2:
                        label = [0., 1., 0.]
                    elif action == 1:
                        label = [1., 0., 0.]

                    if (labels is None):
                        labels = label
                    else:
                        labels = np.vstack((labels, label))

                count += 1
                print ('count: ',  count)
                # move on
                if tick_by_tick:
                    UE.set_ready()
            else:
                UE.reset_env()


    if (args.lstm):
        np.save('rlbot_training_labels_lstm', labels)
        np.save('rlbot_training_dataset_lstm', final_data)
    else:
        np.save('rlbot_training_labels', labels)
        np.save('rlbot_training_dataset', result)

    print 'Finish. ***Stop UE4 game first***'
    print 'Then press any key to exit.'
    ord(getch())
