import argparse
from gym_rlbot.envs.rlbot_training_free import RLBOT_free
from gym_rlbot.envs.rlbot_training_loco import RLBOT_loco

# Hyperparameters
NUM_FRAME = 1000000

parser = argparse.ArgumentParser(description="RLBOT")

# Parse arguments
parser.add_argument("-m", "--mode", type=str, action='store', help="Please specify the mode you wish to run, either train or test", required=True)
parser.add_argument("-t", "--type", type=str, action='store', help="Please specify the type you wish to run, either free or loco", required=True)
parser.add_argument("-l", "--load", type=str, action='store', help="Please specify the file you wish to load weights from(for example saved.h5)", required=False)
parser.add_argument("-s", "--save", type=str, action='store', help="Specify folder to render simulation of network in", required=False)

args = parser.parse_args()
print args

if args.type == "free":
    game_instance = RLBOT_free("DQN")
else:
    game_instance = RLBOT_loco("DQN")

if args.load:
    game_instance.load_network(args.load)

if args.mode == "train":
    game_instance.train(NUM_FRAME)
elif args.mode == "test":
    if args.save:
        stat = game_instance.simulate(path=args.save, save=True)
    else:
        # stat = game_instance.simulateByImage()
        stat = game_instance.simulate()
    print ("Game Statistics")
    print (stat)
