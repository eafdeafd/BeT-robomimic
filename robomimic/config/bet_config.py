"""
Config for BCQ algorithm.
"""

from robomimic.config.base_config import BaseConfig
from robomimic.config.bc_config import BCConfig


class BETConfig(BaseConfig):
    ALGO_NAME = "bet"

    def train_config(self):
        super(BETConfig, self).train_config()
        self.train.seq_length = 10# length of experience sequence to fetch from the buffer
        self.train.batch_size = 100
    def algo_config(self):
        """
        This function populates the `config.algo` attribute of the config, and is given to the 
        `Algo` subclass (see `algo/algo.py`) for each algorithm through the `algo_config` 
        argument to the constructor. Any parameter that an algorithm needs to determine its 
        training and test-time behavior should be populated here.
        """
        # Kmeans 
        self.algo.ae_bins = 32
        self.algo.ae_latent_dim = 1
        self.algo.ae_num_latents = self.algo.ae_bins 
        # self.algo.vocab_size = 1000 # num latents and bins are the same.

        # Optimizer
        self.algo.optim_params.lr = 1e-4
        self.algo.optim_params.wd = .1
        self.algo.optim_params.betas = (.9, .95)
        self.algo.optim_params.grad_norm_clip = 1
        self.algo.optim_params.seed = 42

        # Architecture details
        self.algo.n_layer = 6
        self.algo.n_head = 6
        self.algo.n_embd = 120

        self.algo.window_size = 10
        self.algo.predict_offsets = True
        self.algo.offset_loss_scale = 1000.0
        self.algo.focal_loss_gamma = 2.0
        self.algo.batch_size = 100