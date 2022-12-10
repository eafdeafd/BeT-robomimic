import robomimic
import robomimic.utils.torch_utils as TorchUtils
from robomimic.config import config_factory
from robomimic.scripts.train import train

# make default BC config
config = config_factory(algo_name="bet")


# set config attributes here that you would like to update
config.experiment.name = "bet_multitask"
config.train.data = "/home/andrew/Desktop/robo_sims/robomimic/datasets/square/ph/low_dim.hdf5"
config.train.output_dir = "/home/andrew/Desktop/robo_sims/robomimic/bet_trained_models/bet_multitask"
config.train.batch_size = 100
config.train.num_epochs = 100

# get torch device
device = TorchUtils.get_torch_device(try_to_use_cuda=True)

# launch training run
model = train(config, device=device)
config.train.data = "/home/andrew/Desktop/robo_sims/robomimic/datasets/can/ph/low_dim.hdf5"
config.train.num_epochs = 50

model = train(config, device=device, input_model=model)
config.train.data = "/home/andrew/Desktop/robo_sims/robomimic/datasets/lift/ph/low_dim.hdf5"

model = train(config, device=device, input_model=model)
