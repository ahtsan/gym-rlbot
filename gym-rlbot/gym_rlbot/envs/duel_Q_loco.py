import numpy as np
import keras
from keras.models import load_model, Sequential, Model
from keras.optimizers import Adam
from keras.layers.core import Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.layers import merge, Input, Conv2D
from keras import backend as K
from replay_buffer import ReplayBuffer

# List of hyper-parameters and constants
DECAY_RATE = 0.99
NUM_ACTIONS_ANGULAR = 3
INPUT_WIDTH = 80
INPUT_HEIGHT = 60
# Number of frames to throw into network
NUM_FRAMES = 1

class DuelQ(object):
    """Constructs the desired deep q learning network"""
    def __init__(self):
        self.construct_q_network()
        K.set_image_dim_ordering('th')

    def construct_q_network(self):
        # Uses the network architecture found in DeepMind paper
        self.model = Sequential()
        input_layer = Input(shape = (INPUT_HEIGHT, INPUT_WIDTH, NUM_FRAMES))
        conv1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(input_layer)
        # conv1 = BatchNormalization()(conv1)
        conv2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(conv1)
        # conv2 = BatchNormalization()(conv2)
        conv3 = Conv2D(64, (3, 3), activation = 'relu')(conv2)
        # conv3 = BatchNormalization()(conv3)
        flatten = Flatten()(conv3)
        fc1 = Dense(512)(flatten)
        fc1 = Dropout(0.2)(fc1)
        advantage_angular = Dense(NUM_ACTIONS_ANGULAR)(fc1)
        fc2 = Dense(512)(flatten)
        fc2 = Dropout(0.2)(fc2)
        value = Dense(1)(fc2)
        policy_angular = merge([advantage_angular, value], mode = lambda x: x[0]-K.mean(x[0])+K.tile(x[1], (1,1,NUM_ACTIONS_ANGULAR)), output_shape = (NUM_ACTIONS_ANGULAR,))

        self.model = Model(inputs=[input_layer], outputs=[policy_angular])
        self.model.compile(loss='mse', optimizer=Adam(lr=0.000001))

        self.target_model = Model(inputs=[input_layer], outputs=[policy_angular])
        self.target_model.compile(loss='mse', optimizer=Adam(lr=0.000001))
        print "Successfully constructed networks."

    def predict_movement(self, data, epsilon):
        """Predict movement of game controler where is epsilon
        probability randomly move."""
        angular_q_actions = self.model.predict(data.reshape(1, INPUT_HEIGHT, INPUT_WIDTH, NUM_FRAMES), batch_size = 1)
        opt_policy_angular = np.argmax(angular_q_actions)
        rand_val = np.random.random()
        if rand_val < epsilon:
            opt_policy_angular = np.random.randint(0, NUM_ACTIONS_ANGULAR)
        return opt_policy_angular, angular_q_actions

    def train(self, s_batch, a_batch, r_batch, d_batch, s2_batch, observation_num):
        """Trains network to fit given parameters"""
        batch_size = s_batch.shape[0]
        targets_angular = np.zeros((batch_size, NUM_ACTIONS_ANGULAR))

        for i in xrange(batch_size):
            targets_angular[i] = self.model.predict(s_batch[i].reshape(1, INPUT_HEIGHT, INPUT_WIDTH, NUM_FRAMES), batch_size = 1)
            fut_action_angular = self.target_model.predict(s2_batch[i].reshape(1, INPUT_HEIGHT, INPUT_WIDTH, NUM_FRAMES), batch_size = 1)
            targets_angular[i, a_batch[i]] = r_batch[i]
            if d_batch[i] == False:
                targets_angular[i, a_batch[i]] += DECAY_RATE * np.max(fut_action_angular)

        loss = self.model.train_on_batch(s_batch, [targets_angular])

        # # Print the loss every 10 iterations.
        # if observation_num % 100 == 0:
        #     print "We had a loss equal to ", loss

    def save_network(self, path):
        # Saves model at specified path as h5 file
        self.model.save(path)
        print "Successfully saved network."

    def load_network(self, path):
        self.model.load_weights(path)
        self.target_model.load_weights(path)
        print "Succesfully loaded network."

    def target_train(self):
        model_weights = self.model.get_weights()
        self.target_model.set_weights(model_weights)
        print 'Updated target Q network.'
