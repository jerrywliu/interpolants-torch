# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: icon
#     language: python
#     name: icon
# ---

# %%
import torch
import torch.nn as nn
import torch.optim as optim

torch.set_default_dtype(torch.float64)
import numpy as np

import time
from tqdm import tqdm
import matplotlib.pyplot as plt

import os, sys

BASE_DIR = "/pscratch/sd/j/jwl50/interpolants-torch"
sys.path.append(BASE_DIR)

# %%
from src.models.rational_2d import RationalInterpolation2D
from src.experiments.interpolation.pde_solns.reaction import ReactionTarget
from src.loggers.logger import Logger

# %%
device = "cuda"
rho = 5.0
target = ReactionTarget(rho=rho, device=device)

n_epochs = 10000
base_save_dir = f"/pscratch/sd/j/jwl50/interpolants-torch/plots/interpolation/reaction/rho={rho}"
n_t = 41
n_x = 41
method = "adam"
sample_type = "standard"

# %%
from src.utils.metrics import l2_error, max_error, l2_relative_error

eval_every = 100
n_eval = 200
t_eval = torch.linspace(
    target.domain[0][0],
    target.domain[0][1],
    n_eval,
    device=device,
    requires_grad=True,
)
x_eval = torch.linspace(
    target.domain[1][0],
    target.domain[1][1],
    n_eval + 1,
    device=device,
    requires_grad=True,
)[:-1]

def eval_sampler():
    return t_eval, x_eval

eval_metrics = [l2_error, max_error, l2_relative_error]

# %%
save_dir = os.path.join(base_save_dir, f"rational")
# Logger setup
logger = Logger(path=os.path.join(save_dir, "logger.json"))

# Model setup
bases = ["chebyshev", "chebyshev"]
num_poles = 2
model = RationalInterpolation2D(
    N_1=n_t,
    N_2=n_x,
    bases_1=bases[0],
    domain_1=target.domain[0],
    domain_2=target.domain[1],
    num_poles=num_poles,
    device=device,
)

# # Manually make the weights not learnable
# for i,interpolant in enumerate(model.interpolants):
#     if i != 0:
#         interpolant.values.requires_grad = False
#         interpolant.weights.requires_grad = False
#     else:
#         interpolant.weights.requires_grad=False

# Training setup
optimizer = target.get_optimizer(model, method)

n_t_train = 161
n_x_train = 161

def train_sampler():
    t_nodes = target.sample_domain_1d(
        n_samples=n_t_train,
        dim=0,
        basis="chebyshev",
        type=sample_type,
    )
    x_nodes = target.sample_domain_1d(
        n_samples=n_x_train,
        dim=1,
        basis="chebyshev",
        type=sample_type,
    )
    return [t_nodes, x_nodes]

print(f"Training Rational Interpolant with {method} optimizer...")
target.train(
    model=model,
    n_epochs=n_epochs,
    optimizer=optimizer,
    train_sampler=train_sampler,
    eval_sampler=eval_sampler,
    eval_metrics=eval_metrics,
    eval_every=eval_every,
    save_dir=save_dir,
    logger=logger,
)

# %% [markdown]
# ## Sanity check: manually set the weights and see if we can interpolate

# %%
model = RationalInterpolation2D(
    N_1=n_t,
    N_2=n_x,
    bases_1=bases[0],
    domain_1=target.domain[0],
    domain_2=target.domain[1],
    num_poles=num_poles,
    device=device,
)

# %%
# Set model values
for i in range(len(model.nodes[0])):
    f_i = target.get_function([model.nodes_standard[0][i], model.interpolants[i].nodes_standard])
    model.interpolants[i].values.data = f_i.reshape(-1)

# %%
# Form stack of model values
model_values = torch.stack(
    [model.interpolants[i].values for i in range(len(model.interpolants))],
    dim=0,
)

# %%
target.get_loss(model, model.nodes_standard)
target.plot_solution(model.nodes, model(model.nodes), os.path.join(save_dir, "temp.png"))

# %%
target.plot_solution(model.nodes, model_values, os.path.join(save_dir, "temp2.png"))

# %%
# Form stack of function values(?)
function_values = torch.stack(
    [target.get_function([model.nodes[0][i], model.interpolants[i].nodes]).reshape(-1) for i in range(len(model.interpolants))],
    dim=0,
)
target.plot_solution(model.nodes, function_values, os.path.join(save_dir, "temp3.png"))   

# %%
model.interpolants[0].nodes

# %%
params = [p for p in model.parameters()]
len(params)

# %%
t_eval = torch.linspace(0, 1, 5, device=device)
x_eval = torch.linspace(0, 1, 7, device=device)

# %%
interpolated = torch.stack(
    [model.interpolants[i](x_eval) for i in range(len(model.interpolants))],
    dim=0,
)

# %%
interpolated.shape

# %%
interpolated_2 = model._interpolate_first_dim(t_eval, interpolated)
interpolated_2

# %%
interpolated_2.shape

# %%
