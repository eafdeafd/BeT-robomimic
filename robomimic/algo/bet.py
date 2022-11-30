"""
Implementation of Behavioral Transformers (BeT).
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import robomimic.models.base_nets as BaseNets
import robomimic.models.obs_nets as ObsNets
import robomimic.models.policy_nets as PolicyNets
import robomimic.models.vae_nets as VAENets
import robomimic.utils.loss_utils as LossUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo
from robomimic.algo.bet_models.kmeans import KmeansDiscretizer
from robomimic.algo.bet_models.generators import MinGPT
from robomimic.config import config_factory
import einops
from collections import deque


@register_algo_factory_func("bet")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BeT algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """
    return BET, {}


class BET(PolicyAlgo):
    """
    Normal BET training.
    """
    def __init__(
            self,
            algo_config,
            obs_config,
            global_config,
            obs_key_shapes,
            ac_dim,
            device
    ):
        super().__init__(algo_config, obs_config, global_config, obs_key_shapes, ac_dim, device)
        self.slices = None
        self.window_size = None


    def create_and_seed_discretizer(self, train_loader):
        self.discretizer = KmeansDiscretizer(self.ac_dim)
        self.discretizer.load(train_loader)
        self.discretizer.fit(ncluster=self.algo_config.ae_bins)

    def get_seq_length(self, idx, input):
        count = 0
        for a in input[idx]:
            if 1 >= a >= -1:
                count += 1
        return count
    
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()


        # set up different observation groups for @MIMO_MLP
        input_obs_group_shapes = OrderedDict()
        input_obs_group_shapes["obs"] = OrderedDict(self.obs_shapes)
        self.goal_shapes = OrderedDict()
        output_shapes = OrderedDict(action=(self.ac_dim,))

        assert isinstance(input_obs_group_shapes, OrderedDict)
        assert np.all([isinstance(input_obs_group_shapes[k], OrderedDict) for k in input_obs_group_shapes])
        assert isinstance(output_shapes, OrderedDict)

        self.input_obs_group_shapes = input_obs_group_shapes
        self.output_shapes = output_shapes

        self.nets = nn.ModuleDict()

        encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder)
        # Encoder for all observation groups.
        # For low_dim, does nothing.
        self.nets["encoder"] = ObsNets.ObservationBETGroupEncoder(
            observation_group_shapes=input_obs_group_shapes,
            encoder_kwargs=encoder_kwargs,
        )
        self.nets["ac_encoder"] = ObsNets.ObservationGroupEncoder(
            observation_group_shapes=input_obs_group_shapes,
            encoder_kwargs=encoder_kwargs,
        )

        # flat encoder output dimension
        mlp_input_dim = self.nets["encoder"].output_shape()[0]

        self.nets["mlp"] = BaseNets.MLP(
            input_dim=self.ac_dim,
            output_dim=self.ac_dim,
            layer_dims=[136, 544, 272],
            layer_func=nn.Linear,
            activation=nn.Mish,
            output_activation=nn.Tanh, # make sure non-linearity is applied before decoder
        )


        self.nets["policy"] = MinGPT(
            input_dim=mlp_input_dim,
            latent_dim=self.algo_config.ae_latent_dim,
            vocab_size=self.algo_config.ae_num_latents,
            wd=self.optim_params.wd,
            lr=self.optim_params.lr,
            betas=self.optim_params.betas,
            n_embd = self.algo_config.n_embd,
            predict_offsets=self.algo_config.predict_offsets,
            n_layer=self.algo_config.n_layer,
            n_head=self.algo_config.n_head,
            offset_loss_scale=self.algo_config.offset_loss_scale,
            focal_loss_gamma=self.algo_config.focal_loss_gamma,
            action_dim=self.ac_dim,
            discrete_input=self.algo_config.discrete_input
        )
        self.optim = self.nets["policy"].get_optim()
        self.nets = self.nets.float().to(self.device)
        self.action_encoder = torch.nn.Identity(self.ac_dim)
        self.window_size = self.algo_config.window_size
        self.history = deque(maxlen=self.algo_config.history_size)

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.
        Comes in as N x A 
        Needs to be: N x Slice x A
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training 
        """
        input_batch = batch
        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data. 

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        TODO: Port GPT training code over here
        """
        self.nets["policy"].train()
        with TorchUtils.maybe_no_grad(no_grad=validate), TorchUtils.eval_mode(self.action_encoder):
            #for window in self.window:
            info = super(BET, self).train_on_batch(batch, epoch, validate=validate)

            losses = OrderedDict()
            enc_outputs = self.nets["encoder"](self.algo_config.window_size, **{"obs":batch["obs"]})
            # optional encoding here
            #self.action_encoder(enc_outputs)
            latent = self.discretizer.discretize(batch["actions"])
            sa, so, loss = self.nets["policy"].get_latent_and_loss(obs_rep=enc_outputs, target_latents=latent, return_loss_components=True)

           
            pa = self.discretizer.decode_actions(latent_action_batch=(sa, so))
            mlp_out = self.nets["mlp"](pa)
            
            a_target = batch["actions"]
            actions = mlp_out

            losses["l2_loss"] = nn.MSELoss()(actions, a_target)
            losses["l1_loss"] = nn.SmoothL1Loss()(actions, a_target)
            # cosine direction loss on eef delta position
            losses["cos_loss"] = LossUtils.cosine_loss(actions[..., :3], a_target[..., :3])

            action_losses = [
                self.algo_config.loss.l2_weight * losses["l2_loss"],
                self.algo_config.loss.l1_weight * losses["l1_loss"],
                self.algo_config.loss.cos_weight * losses["cos_loss"],
            ]

            loss += sum(action_losses)
            losses["action_loss"] = loss
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                step_info = OrderedDict()
                
                
                self.optim.zero_grad(set_to_none=True)
                loss.backward()

                # gradient clipping
                torch.nn.utils.clip_grad_norm_(self.nets["policy"].parameters(), self.optim_params.grad_norm_clip)

                # compute grad norms
                grad_norms = 0.
                for p in self.nets["policy"].parameters():
                    # only clip gradients for parameters for which requires_grad is True
                    if p.grad is not None:
                        grad_norms += p.grad.data.norm(2).pow(2).item()

                # step
                self.optim.step()
                
                step_info["policy_grad_norms"] = grad_norms
                info.update(step_info)
            return info
    
    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super(BET, self).log_info(info)
        log["Loss"] = info["losses"]["action_loss"].item()
        if "l2_loss" in info["losses"]:
            log["L2_Loss"] = info["losses"]["l2_loss"].item()
        if "l1_loss" in info["losses"]:
            log["L1_Loss"] = info["losses"]["l1_loss"].item()
        if "cos_loss" in info["losses"]:
            log["Cosine_Loss"] = info["losses"]["cos_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log

    def get_action(self, obs_dict, goal_dict=None):

        """
        Get policy action outputs.
        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        with TorchUtils.eval_mode(self.action_encoder, self.nets["policy"], no_grad=True):
            enc_obs = self.nets["ac_encoder"](**{"obs":obs_dict})
            enc_obs = einops.repeat(enc_obs[0], "obs -> batch obs", batch=1)
            self.history.append(enc_obs)
            enc_obs_seq = torch.stack(tuple(self.history), dim=0)
            latents, offsets = self.nets["policy"].generate_latents(enc_obs_seq)
            action_latents = (latents[:, -1:, :], offsets[:, -1:, :])  # current obs is last obs
            actions = self.discretizer.decode_actions(latent_action_batch=action_latents)
            sampled_action = np.random.randint(len(actions))
            actions = actions[sampled_action]
            actions = einops.rearrange(actions, "1 action_dim -> 1 1 action_dim")
            actions = self.nets["mlp"](actions)
            actions = einops.rearrange(actions, "1 1 action_dim -> action_dim")
            return [actions]
        
    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        self.history = deque(maxlen=self.algo_config.history_size)