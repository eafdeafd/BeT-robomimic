import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

import robomimic.algo.bet_models.libraries.mingpt.model as mingpt_model
import robomimic.algo.bet_models.libraries.mingpt.trainer as mingpt_trainer
from robomimic.algo.bet_models.libraries.loss_fn import FocalLoss, soft_cross_entropy
from robomimic.models.base_nets import Module
from typing import Optional, Tuple

class MinGPT(Module):
    def __init__(
        self,
        input_dim: int,        
        wd: float,
        lr: float,
        betas: Tuple[float, float],
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 768,
        embd_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        block_size: int = 128,
        vocab_size: int = 50257,
        latent_dim: int = 768,  # Ignore, used for compatibility with other models.
        action_dim: int = 0,
        discrete_input: bool = False,
        predict_offsets: bool = False,
        offset_loss_scale: float = 1.0,
        focal_loss_gamma: float = 0.0,

        **kwargs
        ):
        super(MinGPT, self).__init__()
        self.predict_offsets = predict_offsets
        self.action_dim = action_dim
        self.vocab_size = vocab_size
        self.focal_loss_gamma = focal_loss_gamma
        self.offset_loss_scale = offset_loss_scale

        for k, v in kwargs.items():
            setattr(self, k, v)
        config = mingpt_model.GPTConfig(
            input_dim=input_dim,
            vocab_size=vocab_size * (1 + action_dim) if predict_offsets else vocab_size,
            block_size=block_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            discrete_input=discrete_input,
            embd_pdrop=embd_pdrop,
            resid_pdrop=resid_pdrop,
            attn_pdrop=attn_pdrop,
        )
        self.model = mingpt_model.GPT(config)
        self.wd = wd
        self.lr = lr
        self.betas = betas

        
    # In the probabilisitc sense, this model fits and samples from P(latent|observation) given some observation.
    def get_latent_and_loss(
        self,
        obs_rep: torch.Tensor,
        target_latents: torch.Tensor,
        seq_masks: Optional[torch.Tensor] = None,
        return_loss_components: bool = False,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a set of observation representation and generated latents, get the encoded latent and the loss.

        Inputs:
        input_action: Batch of the actions taken in the multimodal demonstrations.
        target_latents: Batch of the latents that the generator should learn to generate the actions from.
        seq_masks: Batch of masks that indicate which timesteps are valid.

        Outputs:
        latent: The sampled latent from the observation.
        loss: The loss of the latent generator.
        """
            # Unlike torch.transformers, GPT takes in batch x seq_len x embd_dim
        # obs_rep = einops.rearrange(obs_rep, "seq batch embed -> batch seq embed")
        # target_latents = einops.rearrange(
        #     target_latents, "seq batch embed -> batch seq embed"
        # )
        # While this has been trained autoregressively,
        # there is no reason why it needs to be so.
        # We can just use the observation as the input and the next latent as the target.

        target_latents, target_offsets = target_latents
        is_soft_target = (target_latents.shape[-1] == self.vocab_size) and (self.vocab_size != 1)
        if is_soft_target:
            target_latents = target_latents.view(-1, target_latents.size(-1))
            criterion = soft_cross_entropy
        else:
            target_latents = target_latents.view(-1)
            if self.vocab_size == 1:
                # unify k-means (target_class == 0) and GMM (target_prob == 1)
                target_latents = torch.zeros_like(target_latents)
            criterion = FocalLoss(gamma=self.focal_loss_gamma)
        output, _ = self.model(obs_rep)
        logits = output[:, :, : self.vocab_size]
        offsets = output[:, :, self.vocab_size :]

        batch = logits.shape[0]
        seq = logits.shape[1]
        offsets = einops.rearrange(
            offsets,
            "N T (V A) -> (N T) V A",  # N = batch, T = seq
            V=self.vocab_size,
            A=self.action_dim,
        )
        # calculate (optionally soft) cross entropy and offset losses
        class_loss = criterion(logits.view(-1, logits.size(-1)), target_latents)
        # offset loss is only calculated on the target class
        # if soft targets, argmax is considered the target class
        selected_offsets = offsets[
            torch.arange(offsets.size(0)),
            target_latents.argmax(dim=-1).view(-1)
            if is_soft_target
            else target_latents.view(-1),
        ]
        offset_loss = self.offset_loss_scale * F.mse_loss(
            selected_offsets, target_offsets.view(-1, self.action_dim)
        )
        loss = offset_loss + class_loss


        # we want to compute action we would've taken also, since need to tanh at the end, similar to generate_latents
        probs = F.softmax(logits, dim=-1)
        batch, seq, choices = probs.shape
        # Sample from the multinomial distribution, one per row.
        sampled_data = torch.multinomial(probs.view(-1, choices), num_samples=1)
        sampled_data = einops.rearrange(
            sampled_data, "(batch seq) 1 -> batch seq 1", batch=batch, seq=seq
        )
        sampled_offsets = offsets[
            torch.arange(offsets.shape[0]), sampled_data.flatten()
        ].view(batch, seq, self.action_dim)

        return sampled_data, sampled_offsets, loss
    def generate_latents(
        self, seq_obses: torch.Tensor) -> torch.Tensor:
        """
        Given a batch of sequences of observations, generate a batch of sequences of latents.

        Inputs:
        seq_obses: Batch of sequences of observations, of shape seq x batch x dim, following the transformer convention.

        Outputs:
        seq_latents: Batch of sequences of latents of shape seq x batch x latent_dim.
        """
        seq, batch, embed = seq_obses.size()
        obs_rep = einops.rearrange(seq_obses, "seq batch embed -> batch seq embed")
        output, _ = self.model(obs_rep, None)
        logits = output[:, :, : self.vocab_size]
        offsets = output[:, :, self.vocab_size :]
        offsets = einops.rearrange(
            offsets,
            "N T (V A) -> (N T) V A",  # N = batch, T = seq
            V=self.vocab_size,
            A=self.action_dim,
        )
        # print("logits", logits, logits.size())
        probs = F.softmax(logits, dim=-1)
        # print("probs", probs, probs.size())
        batch, seq, choices = probs.shape
        # Sample from the multinomial distribution, one per row.
        sampled_data = torch.multinomial(probs.view(-1, choices), num_samples=1)
        # print("sampled_data", sampled_data, sampled_data.size())
        sampled_data = einops.rearrange(
            sampled_data, "(batch seq) 1 -> batch seq 1", batch=batch, seq=seq
        )
        # print("sampled data 2", sampled_data, sampled_data.size())
        sampled_offsets = offsets[
            torch.arange(offsets.shape[0]), sampled_data.flatten()
        ].view(batch, seq, self.action_dim)
        # print("sampled offsets", sampled_offsets, sampled_offsets.size())
        return (sampled_data, sampled_offsets)

    def output_shape(self, input_shape=None):
        return self.vocab_size
    
    def get_optim(self):
        train_config = mingpt_trainer.TrainerConfig(weight_decay=self.wd, learning_rate=self.lr, betas=self.betas)
        return self.model.configure_optimizers(train_config)