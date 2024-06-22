import configargparse
import os
from distutils.util import strtobool

class MARLConfigParser():
    def __init__(self):
        self.parser = configargparse.ArgParser(default_config_files=['experiments/sumo-4x4-independent.config'], 
                                        description="Generate the learning curve for agents trained on the SUMO environment")
        self.parser.add_argument('-c', '--config_path', required=False, is_config_file=True, help='config file path')

        # TODO: remove unecessary configs here, we're just looking at sumo in this file
        # Common arguments
        self.parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                            help='the name of this experiment')
        self.parser.add_argument('--gym-id', type=str, default="CartPole-v0",
                            help='the id of the gym environment')
        self.parser.add_argument('--env-args', type=str, default="",
                            help='string to pass to env init')
        self.parser.add_argument('--learning-rate', type=float, default=7e-4,
                            help='the learning rate of the optimizer')
        self.parser.add_argument('--seed', type=int, default=1,
                            help='seed of the experiment')
        self.parser.add_argument('--total-timesteps', type=int, default=500000,
                            help='total timesteps of the experiments')
        self.parser.add_argument('--max-cycles', type=int, default=100,
                            help='max cycles in each step of the experiments')
        self.parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                            help='if toggled, `torch.backends.cudnn.deterministic=False`')
        self.parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                            help='if toggled, cuda will not be enabled by default')
        self.parser.add_argument('--prod-mode', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                            help='run the script in production mode and use wandb to log outputs')
        self.parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                            help='weather to capture videos of the agent performances (check out `videos` folder)')
        self.parser.add_argument('--wandb-project-name', type=str, default="DA-RL",
                            help="the wandb's project name")
        self.parser.add_argument('--wandb-entity', type=str, default=None,
                            help="the entity (team) of wandb's project")
        self.parser.add_argument('--render', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                            help='if toggled, render environment')
        self.parser.add_argument('--global-obs', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                            help='if toggled, stack agent observations into global state')
        self.parser.add_argument('--gpu-id', type=str, default=None,
                            help='gpu device to use')
        self.parser.add_argument('--nn-save-freq', type=int, default=1000,
                            help='how often to save a copy of the neural network')

        # Algorithm specific arguments
        self.parser.add_argument('--N', type=int, default=3,
                            help='the number of agents')
        self.parser.add_argument('--buffer-size', type=int, default=10000,
                            help='the replay memory buffer size')
        self.parser.add_argument('--gamma', type=float, default=0.99,
                            help='the discount factor gamma')
        self.parser.add_argument('--target-network-frequency', type=int, default=500,
                            help="the timesteps it takes to update the target network")
        self.parser.add_argument('--max-grad-norm', type=float, default=0.5,
                            help='the maximum norm for the gradient clipping')
        self.parser.add_argument('--batch-size', type=int, default=32,
                            help="the batch size of sample from the reply memory")
        self.parser.add_argument('--start-e', type=float, default=1,
                            help="the starting epsilon for exploration")
        self.parser.add_argument('--end-e', type=float, default=0.05,
                            help="the ending epsilon for exploration")
        self.parser.add_argument('--lam', type=float, default=0.01,
                            help="the pension for the variance")
        self.parser.add_argument('--exploration-fraction', type=float, default=0.05,
                            help="the fraction of `total-timesteps` it takes from start-e to go end-e")
        self.parser.add_argument('--learning-starts', type=int, default=10000,
                            help="timestep to start learning")
        self.parser.add_argument('--train-frequency', type=int, default=1,
                            help="the frequency of training")
        self.parser.add_argument('--load-weights', type=bool, default=False,
                        help="whether to load weights for the Q Network")

        # Configuration parameters specific to the SUMO traffic environment
        self.parser.add_argument("--route-file", dest="route", type=str, required=False, help="Route definition xml file.\n")
        self.parser.add_argument("--net-file", dest="net", type=str, required=False, help="Net definition xml file.\n")
        self.parser.add_argument("--mingreen", dest="min_green", type=int, default=10, required=False, help="Minimum time for green lights in SUMO environment.\n")
        self.parser.add_argument("--maxgreen", dest="max_green", type=int, default=30, required=False, help="Maximum time for green lights in SUMO environment.\n")
        self.parser.add_argument("--sumo-gui", dest="sumo_gui", action="store_true", default=False, help="Run with visualization on SUMO (may require firewall permissions).\n")
        self.parser.add_argument("--sumo-seconds", dest="sumo_seconds", type=int, default=10000, required=False, help="Number of simulation seconds. The number of seconds the simulation must end.\n")
        self.parser.add_argument("--sumo-reward", dest="sumo_reward", type=str, default='wait', required=False, help="Reward function: \nThe 'queue'reward returns the negative number of total vehicles stopped at all agents each step, \
                                                                                                                        \nThe 'wait' reward returns the negative number of cummulative seconds that vehicles have been waiting in the episode. \
                                                                                                                        \nThe 'custom-average-speed-limit' reward returns the negative difference between the average speed of all vehicles in the intersection and a provided speed limit, \
                                                                                                                        \nThe 'custom' reward returns the negative sqrt of the difference between the maximum speed of all vehicles in the intersection and a range of allowable speeds.")
        self.parser.add_argument("--sumo-average-speed-limit", dest="sumo_average_speed_limit", type=float, default=7.0, required=False, help="Average speed limit to use if reward function is 'custom-average-speed-limit'\n")
        self.parser.add_argument("--sumo-max-speed-threshold", dest="sumo_max_speed_threshold", type=float, default=13.89, required=False, help="Maximum allowable speed limit to use if reward function is 'custom'\n")
        self.parser.add_argument("--sumo-min-speed-threshold", dest="sumo_min_speed_threshold", type=float, default=1.0, required=False, help="Minimum allowable speed limit to use if reward function is 'custom'\n")

        # Configuration parameters for analyzing sumo env
        self.parser.add_argument("--analysis-steps", dest="analysis_steps", type=int, default=500, required=True, 
                            help="The number of time steps at which we want to investigate the perfomance of the algorithm. E.g. display how the training was going at the 10,000 checkpoint. Note there must be a nn .pt file for each agent at this step.\n")
        self.parser.add_argument("--analysis-training-round", dest="analysis_training_round", type=int, default=10, required=False, 
                            help="The training round from batch offline RL at which we want to investigate the perfomance of the algorithm. E.g. display how the policies were after 5 rounds of training. Note there must be a nn .pt file for each agent at this training round.\n")        
        self.parser.add_argument("--nn-directory", dest="nn_directory", type=str, default=None, required=False, 
                            help="The directory containing the nn .pt files to load for analysis.\n")
        self.parser.add_argument("--nn-queue-directory", dest="nn_queue_directory", type=str, default=None, required=True, 
                            help="The directory containing the nn .pt files from the queue model policies to use for generating a dataset.\n")
        self.parser.add_argument("--nn-speed-overage-directory", dest="nn_speed_overage_directory", type=str, default=None, required=True, 
                            help="The directory containing the nn .pt files from the speed overage model policies to use for generating a dataset.\n")
        self.parser.add_argument("--parameter-sharing-model", dest="parameter_sharing_model", action="store_true", default=False, required=False, 
                            help="Flag indicating if the model trained leveraged parameter sharing or not (needed to identify the size of the model to load).\n")
        self.parser.add_argument("--use-true-value-functions", dest="use_true_value_functions", type=lambda x:bool(strtobool(x)), default=False, required=False, 
                            help="Flag indicating if true value functions should be ingested for the experiment.\n")
        self.parser.add_argument("--nn-true-g1-dir", dest="nn_true_g1_dir", type=str, default="", required=False, 
                            help="Directory containing the nn .pt files to load as the 'true' G1 constraint value functions.\n")
        self.parser.add_argument("--nn-true-g2-dir", dest="nn_true_g2_dir", type=str, default="", required=False, 
                            help="Directory containing the nn .pt files to load as the 'true' G2 constraint value functions.\n")
        self.parser.add_argument("--dataset-path", dest="dataset_path", type=str, default="", required=False, 
                            help="Path to previously generated dataset .pkl file.\n")

    def parse_args(self):
        return self.parser.parse_args()