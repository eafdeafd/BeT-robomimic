See https://robomimic.github.io/ for the simulation framework. \
The core BeT implementation for robomimic is contained in bet.py and utilizes mingpt, generators, and kmeans files. \
To train, normal robomimic training commands. For example to train BeT on lift's proficient human dataset, python robomimic/scripts/train.py --config robomimic/exps/templates/bet.json --dataset datasets/can/ph/low_dim.hdf5 \
For transfer learning, edit the transfer_learning script in robomimic/scripts and run: python transfer_learning.py\
For multitask learning, edit the multi_task training script in robomimic/scripts to your liking and run: python robomimic/scripts/multitask_train.py --config robomimic/exps/templates/bet_multi.json --dataset datasets/square/mh/low_dim.hdf5 \

Current multi-task functionality trains on Lift, PickPlaceCan, and NutAssemblySquare Multi-Human datasets. \
