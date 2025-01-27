# Script to run PINNs experiments using MLPs

import argparse
import os
import sys

# Training setup
method = "adam"
sample_type = "uniform"
n_epochs = 1000000
cuda_device = 0

# Evaluation setup
eval_every = 1000

# MLP
n_layers = 3
hidden_dim = 256

# Experiments
experiment_list = [
    "advection",
    "reaction",
    "wave",
    "burgers",
    "allen_cahn",
]

experiment_name_to_cmd = {
    "advection": f"CUDA_VISIBLE_DEVICES={cuda_device} python -m src.experiments.pdes.simple.advection --c 40 --n_epochs {n_epochs} --method {method} --sample_type {sample_type} --n_layers {n_layers} --hidden_dim {hidden_dim} --model mlp --eval_every {eval_every}",
    "reaction": f"CUDA_VISIBLE_DEVICES={cuda_device} python -m src.experiments.pdes.simple.reaction --rho 5 --n_epochs {n_epochs} --method {method} --sample_type {sample_type} --n_layers {n_layers} --hidden_dim {hidden_dim} --model mlp --eval_every {eval_every}",
    "wave": f"CUDA_VISIBLE_DEVICES={cuda_device} python -m src.experiments.pdes.simple.wave --c 2 --beta 5 --n_epochs {n_epochs} --method {method} --sample_type {sample_type} --n_layers {n_layers} --hidden_dim {hidden_dim} --model mlp --eval_every {eval_every}",
    "burgers": f"CUDA_VISIBLE_DEVICES={cuda_device} python -m src.experiments.pdes.benchmarks.burgers --n_epochs {n_epochs} --method {method} --sample_type {sample_type} --n_layers {n_layers} --hidden_dim {hidden_dim} --model mlp --eval_every {eval_every}",
    "allen_cahn": f"CUDA_VISIBLE_DEVICES={cuda_device} python -m src.experiments.pdes.benchmarks.allen_cahn --n_epochs {n_epochs} --method {method} --sample_type {sample_type} --n_layers {n_layers} --hidden_dim {hidden_dim} --model mlp --eval_every {eval_every}",
}

for experiment in experiment_list:
    print(f"Running {experiment}...")
    os.system(experiment_name_to_cmd[experiment])
