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

from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt

# %%
import os, sys

BASE_DIR = "/pscratch/sd/j/jwl50/interpolants-torch"
sys.path.append(BASE_DIR)

# %%
from src.models.interpolant_nd import SpectralInterpolationND

# %%
