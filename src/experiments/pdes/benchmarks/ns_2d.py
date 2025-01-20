import matplotlib.pyplot as plt
import os
from time import time
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import List, Callable

from src.experiments.pdes.base_pde import BasePDE
from src.models.interpolant_nd import SpectralInterpolationND

"""
2D incompressible Navier-Stokes equation: TODO
"""
