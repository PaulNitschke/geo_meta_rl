import time
from tqdm import tqdm

import wandb
import torch as th

"""
Classes and functions for initialization for hereditary symmetry discovery.
ExponentialLinearRegressor: intializes left-actions.
identity_init_neural_net: initializes a neural network to the identity (used for encoder and decoder).
"""


class ExponentialLinearRegressor(th.nn.Module):
    """
    # Learns a matrix W such that exp(W) \cdot X ≈ Y.
    """
    def __init__(self, input_dim: int, seed:int, log_wandb:bool=True):
        super().__init__()
        th.manual_seed(seed)
        self.W = th.nn.Parameter(th.randn(input_dim, input_dim))
        self.log_wandb = log_wandb

    def forward(self, X: th.Tensor) -> th.Tensor:
        """
        X: (N, n) input matrix
        Returns: (N, n) prediction
        """
        W_exp = th.matrix_exp(self.W)
        return (W_exp @ X.T).T  # Apply from left, return (N, n)

    def loss(self, X: th.Tensor, Y: th.Tensor) -> th.Tensor:
        """
        Computes MSE loss between predicted and true outputs.
        """
        Y_pred = self.forward(X)
        return th.mean((Y - Y_pred) ** 2)

    def fit(self, X: th.Tensor, Y: th.Tensor, lr: float = 1e-2,
            epochs: int = 1000, verbose: bool = False):
        """
        Fits the model parameters to minimize MSE between exp(W) * X and Y.
        """
        optimizer = th.optim.Adam([self.W], lr=lr)
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = self.loss(X, Y)
            loss.backward()
            optimizer.step()
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
            if self.log_wandb and (epoch % 50 == 0):
                wandb.log({"init/log_left_actions": loss.item()})
                time.sleep(0.05)     
        return self.W.data.clone()
    

def identity_init_neural_net(network: callable, 
                             tasks_ps: list,
                             name:str="network",
                             log_wandb:bool=False,
                             n_steps: int = 5_000):
    """Initializes a neural network to the identity map on all tasks via gradient flow."""

    def stack_samples(ps: list):
        """Stacks samples from all tasks into a single tensor."""
        _n_tasks = len(tasks_ps)
        _n_samples_per_task, ambient_dim = tasks_ps[0].shape
        ps = th.empty([_n_tasks, _n_samples_per_task, ambient_dim], dtype=th.float32)
        for i, task_ps in enumerate(tasks_ps):
            ps[i] = task_ps
        return ps.reshape([-1, ambient_dim])

    ps = stack_samples(tasks_ps)

    opt = th.optim.Adam(network.parameters(), lr=1e-3)

    pbar = tqdm(range(n_steps), desc="Initializing to identity")

    for step in pbar:
        opt.zero_grad()

        encoded_ps = network(ps)

        loss = th.nn.functional.mse_loss(encoded_ps, ps)
        loss.backward()

        opt.step()

        if step%100==0:
            pbar.set_postfix({
                "total": f"{loss.item():.4e}"
            })

        if log_wandb and step % 100 == 0:
            wandb.log({f"init/identity_{name}": loss.item()})
            time.sleep(0.05)     
    return network