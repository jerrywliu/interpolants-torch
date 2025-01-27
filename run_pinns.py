# Script to run PINNs experiments using MLPs

import argparse
import os
import sys

args = argparse.ArgumentParser()
args.add_argument("--method", type=str, default="adam")
args.add_argument("--sample_type", type=str, default="uniform")
args.add_argument("--n_epochs", type=int, default=1000000)
args.add_argument("--cuda_device", type=int, default=0)
args.add_argument("--eval_every", type=int, default=1000)
args.add_argument("--n_layers", type=int, default=3)
args.add_argument("--hidden_dim", type=int, default=256)
args = args.parse_args()

# Training setup
method = args.method
sample_type = args.sample_type
n_epochs = args.n_epochs
cuda_device = args.cuda_device

# Evaluation setup
eval_every = args.eval_every

# MLP
n_layers = args.n_layers
hidden_dim = args.hidden_dim

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
