import numpy as np
import random

# input image dimensions
img_rows, img_cols = 100, 100

def load_data():

    all_images = np.load('rlbot_training_dataset.npy')
    print ('All images: ' + str(all_images.shape))
    #all_images = all_images.reshape(all_images.shape[0], img_rows, img_cols, 1)

    all_labels = np.load('rlbot_training_labels.npy')
    print ('All labels: ' + str(all_labels.shape))

    percent_train = 80
    print (str(percent_train) + ' percent of data is for training.')

    mask_train, mask_test = [], []
    number_of_sample = all_images.shape[0]
    number_of_train_sample = int(number_of_sample*percent_train/100)
    number_of_test_sample = number_of_sample - int(number_of_sample*percent_train/100)

    for i in range(0, number_of_sample):
        eps = random.randint(1, 10)
        if eps <= int(percent_train/10) and len(mask_train) < number_of_train_sample:
            mask_train.append(i)
        elif len(mask_test) < number_of_test_sample:
            mask_test.append(i)
        else:
            mask_train.append(i)

    train_images = all_images[mask_train]
    train_labels = all_labels[mask_train]

    test_images = all_images[mask_test]
    test_labels = all_labels[mask_test]

    print ('Number of training samples: ' + str(train_images.shape[0]))
    print ('Number of testing samples: ' + str(test_images.shape[0]))

    return (train_images, train_labels), (test_images, test_labels)

if __name__ == "__main__":
    load_data()
