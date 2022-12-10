import torch
import numpy as np
import tqdm
from typing import Optional, Tuple, Union
import einops

class KmeansDiscretizer():
    """
    Need own implementation of K-means for residual action correction
    TODO: switch to einops operations
    """

    def __init__(self, action_dim : int, device: Union[str, torch.device] = "cuda") -> None:
        self.action_dim = action_dim
        self.device = device
        self.all_actions = None

    def load(self, train_loader):
        all_actions = []
        if type(train_loader) == list:
            for loader in train_loader:
                for d in loader:
                    action = d["actions"]
                    for a in action:
                        all_actions.append(a[0])
        else:
            for d in train_loader:
                action = d["actions"]
                for a in action:
                    all_actions.append(a[0])
        self.all_actions = torch.stack(all_actions)

    def fit(self, niter: int = 100, ncluster: int = 512) -> any:
        assert self.all_actions is not None
        actions = self.all_actions
        self.action_dim = actions.shape[-1]
        x = actions.view(-1, self.action_dim)
        self.nbins = ncluster
        """
        Simple k-means clustering algorithm adapted from Karpathy's minGPT libary
        https://github.com/karpathy/minGPT/blob/master/play_image.ipynb
        """
        N, D = x.size()
        c = x[torch.randperm(N)[:ncluster]]  # init clusters at random

        pbar = tqdm.trange(niter)
        pbar.set_description("K-means clustering")
        for i in pbar:
            # assign all pixels to the closest codebook element
            a = ((x[:, None, :] - c[None, :, :]) ** 2).sum(-1).argmin(1)
            # move each codebook element to be the mean of the pixels that assigned to it
            c = torch.stack([x[a == k].mean(0) for k in range(ncluster)])
            # re-assign any poorly positioned codebook elements
            nanix = torch.any(torch.isnan(c), dim=1)
            ndead = nanix.sum().item()
            if ndead:
                tqdm.tqdm.write(
                    "done step %d/%d, re-initialized %d dead clusters"
                    % (i + 1, niter, ndead)
                )
            c[nanix] = x[torch.randperm(N)[:ndead]]  # re-init dead clusters
        self.bins = c.to(self.device)
        return c

    def discretize(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given the input action, discretize it using the k-Means clustering algorithm.

        Inputs:
        input_action (shape: ... x action_dim): The input action to discretize. This can be in a batch,
        and is generally assumed that the last dimnesion is the action dimension.

        Outputs:
        discretized_action (shape: ... x num_tokens): The discretized action.
        If self.predict_offsets is True, then the offsets are also returned.
        """
        assert (
                    actions.shape[-1] == self.action_dim
                )        
        flattened_actions = actions.view(-1, self.action_dim)
        closest_clusters = torch.argmin(torch.sum((flattened_actions[:, None, :] - self.bins[None, :, :]) ** 2, dim=2), dim=1)
        discrete_actions = closest_clusters.view(actions.shape[:-1] + (1,))
        
        reconstructed_action = self.decode_actions(discrete_actions)
        offsets = actions - reconstructed_action
        return (discrete_actions, offsets)

    def decode_actions(self, latent_action_batch):
        """
        Given the latent action, reconstruct the original action.

        Inputs:
        latent_action (shape: ... x 1): The latent action to reconstruct. This can be in a batch,
        and is generally assumed that the last dimension is the action dimension. If the latent_action_batch
        is a tuple, then it is assumed to be (discretized_action, offsets).

        Outputs:
        reconstructed_action (shape: ... x action_dim): The reconstructed action.
        """
        offsets = None
        if type(latent_action_batch) == tuple:
            latent_action_batch, offsets = latent_action_batch
        # get the closest cluster center
        closest_cluster_center = self.bins[latent_action_batch]
        # Reshape to the original shape
        reconstructed_action = closest_cluster_center.view(
            latent_action_batch.shape[:-1] + (self.action_dim,)
        )
        if offsets is not None:
            reconstructed_action += offsets
        return reconstructed_action