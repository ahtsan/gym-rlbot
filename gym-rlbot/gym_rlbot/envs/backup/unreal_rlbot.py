import gym
import cv2
from replay_buffer import ReplayBuffer
import numpy as np
from duel_Q import DuelQ
from deep_Q import DeepQ
import time
from gym import wrappers
import os

LOG_NAME_SAVE = 'logs'
MONITOR_DIR = LOG_NAME_SAVE + '/monitor/' #the path to save monitor file

# List of hyper-parameters and constants
BUFFER_SIZE = 50000
MINIBATCH_SIZE = 16
EPSILON_DECAY = 20000
MIN_OBSERVATION = 100
FINAL_EPSILON = 0.02
INITIAL_EPSILON = 1.0
# Number of frames to throw into network
NUM_FRAMES = 4
SAVE_PER_EPOCH = 5000
INPUT_WIDTH = 80
INPUT_HEIGHT = 60

class RLBOT(object):

    def __init__(self, mode):
        print ('Creating gym environment...')
        self.env = gym.make('rlbot-v0')
        print ('Created gym environment.')
        # init log file
        if not os.path.exists(MONITOR_DIR):
            os.makedirs(MONITOR_DIR)

        self.env = wrappers.Monitor(self.env, MONITOR_DIR + 'tmp', write_upon_reset=True, force=True)
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)

        # Construct appropriate network based on flags
        print ('Constructing networks...')
        if mode == "DDQN":
            self.deep_q = DeepQ()
        elif mode == "DQN":
            self.deep_q = DuelQ()

        # A buffer that keeps the last 3 images
        self.process_buffer = []

    def load_network(self, path):
        self.deep_q.load_network(path)

    def get_state(self):
        """Converts the list of NUM_FRAMES images in the process buffer
        into one training sample"""
        # black_buffer = map(lambda x: cv2.resize(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), (84, 90)), self.process_buffer)
        # black_buffer = map(lambda x: x[1:85, :, np.newaxis], self.process_buffer)
        if (NUM_FRAMES > 1):
            while (len(self.process_buffer) < NUM_FRAMES):
                s, _, _, _ = self.env.step(-1)
                self.process_buffer.append(s)

            return np.concatenate(self.process_buffer, axis=2)
        else:
            return self.process_buffer

    def train(self, num_frames):
        observation_num = 0
        epsilon = INITIAL_EPSILON
        alive_frame = 0
        total_reward = 0

        self.env.reset()
        curr_state = self.get_state()
        predict_movement, predict_q_values = self.deep_q.predict_movement(curr_state, epsilon)

        print ('Start!')
        while observation_num < num_frames:

            # Slowly decay the learning rate
            if epsilon > FINAL_EPSILON and self.replay_buffer.size() >= MIN_OBSERVATION:
                epsilon -= (INITIAL_EPSILON-FINAL_EPSILON)/EPSILON_DECAY

            curr_state = self.get_state()
            predict_movement, predict_q_values = self.deep_q.predict_movement(curr_state, epsilon)

            temp_observation, temp_reward, done, _ = self.env.step(predict_movement)
            self.process_buffer = self.process_buffer[1:]
            self.process_buffer.append(temp_observation)

            total_reward += temp_reward

            new_state = self.get_state()
            self.replay_buffer.add(curr_state, predict_movement, temp_reward, done, new_state)

            if done:
                # total_reward += 0.01 * alive_frame
                print "EPS: ", observation_num, ". EPSILON: ", epsilon, ". Lived time: ", alive_frame, ". Total reward: ", total_reward
                self.env.reset()
                self.process_buffer = []
                alive_frame = 0
                total_reward = 0

            if self.replay_buffer.size() >= MIN_OBSERVATION:
                if self.replay_buffer.size() == MIN_OBSERVATION:
                    print ('Start training.')
                s_batch, a_batch, r_batch, d_batch, s2_batch = self.replay_buffer.sample(MINIBATCH_SIZE)
                self.deep_q.train(s_batch, a_batch, r_batch, d_batch, s2_batch, observation_num)
                self.deep_q.target_train()

            # Save the network every 10000 iterations
            if observation_num % SAVE_PER_EPOCH == 0:
                print ("Saving Network at " + str(observation_num))
                self.deep_q.save_network("saved_"+str(observation_num)+".h5")

            alive_frame += 1
            observation_num += 1

    def simulate(self, path = "", save = False):
        """Simulates game"""
        total_attempt = 100
        reward_list = []
        done = False
        tot_award = 0
        attempt = 0
        if save:
            self.env.monitor.start(path, force=True)
        self.env.reset()
        curr_state = self.get_state()
        predict_movement, predict_q_values = self.deep_q.predict_movement(curr_state, 0)

        print ('Start simulation')
        while attempt < total_attempt:
            tot_award = 0
            print ('Attempt ' + str(attempt) + "/" + str(total_attempt))
            while not done:
                state = self.get_state()
                predict_movement, predict_q_values = self.deep_q.predict_movement(state, 0)
                observation, reward, done, info = self.env.step(predict_movement)
                tot_award += reward
                self.process_buffer = self.process_buffer[1:]
                self.process_buffer.append(observation)
            self.env.reset()
            if save:
                self.env.monitor.close()
            print ("Successful: " + str(info['Arrival']) + ". Reward: " + str(tot_award))
            reward_list.append(tot_award)
            done = False
            self.process_buffer = []
            attempt += 1
        self.env.step(-2)

        return np.mean(reward_list), np.std(reward_list)
