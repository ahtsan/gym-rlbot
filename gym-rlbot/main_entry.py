import argparse
from gym_rlbot.envs.unreal_rlbot import RLBOT
from gym_rlbot.envs.unreal_rlbot_new import RLBOT_new

# Hyperparameters
NUM_FRAME = 1000000

parser = argparse.ArgumentParser(description="Train and test different networks on Space Invaders")

# Parse arguments
# parser.add_argument("-n", "--network", type=str, action='store', help="Please specify the network you wish to use, either DQN or DDQN", required=True)
parser.add_argument("-m", "--mode", type=str, action='store', help="Please specify the mode you wish to run, either train or test", required=True)
parser.add_argument("-l", "--load", type=str, action='store', help="Please specify the file you wish to load weights from(for example saved.h5)", required=False)
parser.add_argument("-s", "--save", type=str, action='store', help="Specify folder to render simulation of network in", required=False)
parser.add_argument("-v", "--view", action='store_true', help="Display the network playing a game of space-invaders. Is overriden by the -s command", required=False)

args = parser.parse_args()
print args

# game_instance = RLBOT_new("DQN")
game_instance = RLBOT_new("DQN")

if args.load:
    game_instance.load_network(args.load)

if args.mode == "train":
    game_instance.train(NUM_FRAME)
elif args.mode == "test":
    if args.save:
        stat = game_instance.simulate(path=args.save, save=True)
    else:
        stat = game_instance.simulate()
    print ("Game Statistics")
    print (stat)
