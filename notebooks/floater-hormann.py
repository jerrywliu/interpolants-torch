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

# %% [markdown]
# ## In this notebook, we try out the family of barycentric rational interpolations proposed by Floater and Hormann, 2007.

# %% [markdown]
# We start from the barycentric rational interpolants. Given nodes $\{x_i\}_{i=0}^N$ and function values $f_i = f(x_i)$, a barycentric rational interpolant has the form
# $$\frac{\sum_{k=0}^N \frac{w_k f_k}{x-x_k}}{\sum_{k=0}^N \frac{w_k}{x-x_k}}$$
# for some choice of barycentric weights $\{w_i\}_{i=0}^N$.
#
# Note this is a rational function with numerator and denominator having degrees at most $N$. In fact, all such rational functions that interpolate the data can be written in this form.

# %% [markdown]
# In the special case of $w_i = \prod_{j=0, j \neq i}^N \frac{1}{x_i - x_j}$, the interpolant reduces to the unique polynomial interpolant.

# %% [markdown]
# The family of Floater-Hormann barycentric rationals have a parameter $d$ which controls how "local" the interpolants are. Specifically, the rational interpolants are written as a weighted combination of local polynomial interpolants which each interpolate $d$ points. Setting $d = n$, we recover the polynomial interpolants.
#
# The barycentric weights as a function of $d$ are:
# $$w_i = \sum_{k=\min(0, i-d)}^{\max(n-d, i)} (-1)^k \prod_{j=i, \neq k}^{n} \frac{1}{x_i - x_j}$$

# %%
import torch
import torch.nn as nn
import torch.optim as optim

torch.set_default_dtype(torch.float64)
import numpy as np

import time
from tqdm import tqdm
import matplotlib.pyplot as plt

# %%
import os, sys

BASE_DIR = "/pscratch/sd/j/jwl50/interpolants-torch"
sys.path.append(BASE_DIR)

# %%
from src.models.floater_hormann import RationalInterpolation1D, get_cheb_nodes

# %%
from src.experiments.interpolation.simple_fcns.abs_1d import Abs1DTarget
from src.experiments.interpolation.simple_fcns.logistic_1d import Logistic1DTarget


# %%
def plot_helper(
    train_points,
    eval_points,
    train_values,
    eval_values_pred,
    eval_values_true,
    title=None
):
    # Create a figure with 1x2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    if title:
        fig.suptitle(title)

    # Left subplot: True vs Predicted
    ax1.plot(eval_points, eval_values_true, "k:", label="True function")
    ax1.plot(eval_points, eval_values_pred, "b-", label="Predicted")
    ax1.plot(train_points, train_values, "ro", label="Training points")
    ax1.set_title("True vs Predicted")
    ax1.legend()
    ax1.grid(True)

    # Right subplot: Error (log scale)
    errors = torch.abs(eval_values_pred - eval_values_true)
    max_error = torch.max(errors).item()
    ax2.semilogy(eval_points, errors, "b-", label="Absolute Error")
    ax2.set_title(f"Absolute Error. Max = {max_error:1.3e}")
    ax2.legend()
    ax2.grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    plt.show()


# %% [markdown]
# ## $|x|$

# %%
# Target
target = Abs1DTarget()
n = 51
n_eval = 200
x_eval = torch.linspace(target.domain[0][0], target.domain[0][1], n_eval)
f_eval = target.get_function(x_eval)

# %%
# Model
for d in [1, 2, 4, 8, 16, 32, n-1]:
    # Set data
    model = RationalInterpolation1D(nodes=get_cheb_nodes(n), domain=target.domain[0], weights_type="floater_hormann", floater_hormann_d=d)
    x_train = model.nodes
    f_train = target.get_function(x_train)
    model.values.data = f_train

    # Interpolate
    f_eval_pred = model(x_eval).detach()
    plot_helper(x_train, x_eval, f_train, f_eval_pred, f_eval, title=f"Floater-Hormann, d={d}")

# %% [markdown]
# ## Logistic function (sharp)

# %%
# Target
target = Logistic1DTarget()
n = 51
n_eval = 200
x_eval = torch.linspace(target.domain[0][0], target.domain[0][1], n_eval)
f_eval = target.get_function(x_eval)

# %%
# Model
for d in [1, 2, 4, 8, 16, 32, n-1]:
    # Set data
    model = RationalInterpolation1D(nodes=get_cheb_nodes(n), domain=target.domain[0], weights_type="floater_hormann", floater_hormann_d=d)
    x_train = model.nodes
    f_train = target.get_function(x_train)
    model.values.data = f_train

    # Interpolate
    f_eval_pred = model(x_eval).detach()
    plot_helper(x_train, x_eval, f_train, f_eval_pred, f_eval, title=f"Floater-Hormann, d={d}")

# %% [markdown]
# # Show scaling

# %%
from src.utils.metrics import l2_error, max_error, l2_relative_error
import math

# %%
# Target
target = Logistic1DTarget()
ns = [11, 21, 41, 81, 161, 321]
n_eval = 200
x_eval = torch.linspace(target.domain[0][0], target.domain[0][1], n_eval)
f_eval = target.get_function(x_eval)

# Model
for n in ns:
    ds = [2**i for i in range(math.floor(math.log2(n))+1)] + [n-1]
    errors = []
    for d in ds:
        # Set data
        model = RationalInterpolation1D(nodes=get_cheb_nodes(n), domain=target.domain[0], weights_type="floater_hormann", floater_hormann_d=d)
        x_train = model.nodes
        f_train = target.get_function(x_train)
        model.values.data = f_train
    
        # Interpolate
        f_eval_pred = model(x_eval).detach()
        error = l2_relative_error(f_eval_pred, f_eval)
        errors.append(error)
    plt.plot([d/(n-1) for d in ds], errors, "o-", label=n)
plt.xlabel("d / n")
plt.ylabel("error")
plt.yscale("log")
plt.title("RMSE, Floater-Hormann on logistic function, Cheb points")
plt.legend()
plt.show()

# %%
# Target
target = Logistic1DTarget()
ns = [11, 21, 41, 81, 161, 321]
n_eval = 200
x_eval = torch.linspace(target.domain[0][0], target.domain[0][1], n_eval)
f_eval = target.get_function(x_eval)

# Model
for n in ns:
    ds = [2**i for i in range(math.floor(math.log2(n))+1)] + [n-1]
    errors = []
    for d in ds:
        # Set data
        model = RationalInterpolation1D(nodes=get_cheb_nodes(n), domain=target.domain[0], weights_type="floater_hormann", floater_hormann_d=d)
        x_train = model.nodes
        f_train = target.get_function(x_train)
        model.values.data = f_train
    
        # Interpolate
        f_eval_pred = model(x_eval).detach()
        error = max_error(f_eval_pred, f_eval)
        errors.append(error)
    plt.plot([d/(n-1) for d in ds], errors, "o-", label=n)
plt.xlabel("d / n")
plt.ylabel("error")
plt.yscale("log")
plt.title("$L_{\infty}$, Floater-Hormann on logistic function, Cheb points")
plt.legend()
plt.show()

# %% [markdown]
# ## Change the point distribution

# %% [markdown]
# ### Equispaced

# %%
# Target
target = Logistic1DTarget()
n = 41
n_eval = 200
x_eval = torch.linspace(target.domain[0][0], target.domain[0][1], n_eval)
f_eval = target.get_function(x_eval)

# Model
nodes = torch.linspace(-1, 1, n)
for d in [1, 2, 4]:
    # Set data
    model = RationalInterpolation1D(nodes=nodes, domain=target.domain[0], weights_type="floater_hormann", floater_hormann_d=d)
    x_train = model.nodes
    f_train = target.get_function(x_train)
    model.values.data = f_train

    # Interpolate
    f_eval_pred = model(x_eval).detach()
    plot_helper(x_train, x_eval, f_train, f_eval_pred, f_eval, title=f"Floater-Hormann, d={d}")

# %%
# Target
target = Logistic1DTarget()
ns = [21, 41, 81, 161, 321]
n_eval = 200
x_eval = torch.linspace(target.domain[0][0], target.domain[0][1], n_eval)
f_eval = target.get_function(x_eval)

# Model
for n in ns:
    nodes = torch.linspace(-1, 1, n)
    ds = [2**i for i in range(math.floor(math.log2(n))+1)] + [n-1]
    errors = []
    for d in ds:
        # Set data
        model = RationalInterpolation1D(nodes=nodes, domain=target.domain[0], weights_type="floater_hormann", floater_hormann_d=d)
        x_train = model.nodes
        f_train = target.get_function(x_train)
        model.values.data = f_train
    
        # Interpolate
        f_eval_pred = model(x_eval).detach()
        error = max_error(f_eval_pred, f_eval)
        errors.append(error)
    plt.plot([d/(n-1) for d in ds], errors, "o-", label=n)
plt.xlabel("d / n")
plt.ylabel("error")
plt.yscale("log")
plt.title("$L_{\infty}$, Floater-Hormann on logistic function, equispaced points")
plt.legend()
plt.show()

# %% [markdown]
# ### Sample more around the jump

# %%
# Target
target = Logistic1DTarget()
n = 41
n_eval = 200
x_eval = torch.linspace(target.domain[0][0], target.domain[0][1], n_eval)
f_eval = target.get_function(x_eval)

# Model
nodes = [get_cheb_nodes(n//2+1), torch.linspace(-0.6, -0.4, n//4), torch.linspace(0.4, 0.6, n//4)]
# nodes = [get_cheb_nodes(n//2+1), torch.linspace(-0.6, 0.6, n//2)]
nodes = torch.cat(nodes)
for d in [1, 2, 3, 4]:
    # Set data
    model = RationalInterpolation1D(nodes=nodes, domain=target.domain[0], weights_type="floater_hormann", floater_hormann_d=d)
    x_train = model.nodes
    f_train = target.get_function(x_train)
    model.values.data = f_train

    # Interpolate
    f_eval_pred = model(x_eval).detach()
    plot_helper(x_train, x_eval, f_train, f_eval_pred, f_eval, title=f"Floater-Hormann, d={d}")

# %%
# Target
target = Logistic1DTarget()
ns = [21, 41, 81, 161, 321]
n_eval = 200
x_eval = torch.linspace(target.domain[0][0], target.domain[0][1], n_eval)
f_eval = target.get_function(x_eval)

# Model
for n in ns:
    nodes = [get_cheb_nodes(n//2+1), torch.linspace(-0.6, -0.4, n//4), torch.linspace(0.4, 0.6, n//4)]
    nodes = torch.cat(nodes)
    ds = [2**i for i in range(math.floor(math.log2(n))+1)] + [n-1]
    errors = []
    for d in ds:
        # Set data
        model = RationalInterpolation1D(nodes=nodes, domain=target.domain[0], weights_type="floater_hormann", floater_hormann_d=d)
        x_train = model.nodes
        f_train = target.get_function(x_train)
        model.values.data = f_train
    
        # Interpolate
        f_eval_pred = model(x_eval).detach()
        error = max_error(f_eval_pred, f_eval)
        errors.append(error)
    plt.plot([d/(n-1) for d in ds], errors, "o-", label=n)
plt.xlabel("d / n")
plt.ylabel("error")
plt.yscale("log")
plt.title("$L_{\infty}$, Floater-Hormann on logistic function, Cheb points")
plt.legend()
plt.show()

# %% [markdown]
# # Main takeaways from these experiments:
# - For small $d$, these interpolants are surprisingly good at fitting functions with near-discontinuities, like the logistic function. The convergence rate is about the same as for polynomial interpolants with Chebyshev nodes. This is true for the rational interpolant on Chebyshev nodes but also for equispaced, interestingly, unlike polynomials.
# - Unfortunately, this family of rationals isn't flexible enough to capture the near-discontinuities in an efficient way. Specifically if we oversample near the discontinuity, we see the Runge phenomenon. This makes sense since these rationals are equivalent to a linear combination of polynomial interpolants.
# - Seems we'll need to properly learn the pole placements to beat polynomial convergence rates.

# %%
