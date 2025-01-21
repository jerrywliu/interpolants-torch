#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --account=m1266
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --module=gpu,nccl-2.15

conda activate /pscratch/sd/j/jwl50/interpolants-torch/.env

# PDEs
# Advection
# python -m src.experiments.pdes.simple.advection --c 8 --nt 9 --nx 8
python -m src.experiments.pdes.simple.advection --c 40 --nt 81 --nx 80 --n_epochs 500000
# python -m src.experiments.pdes.simple.advection --c 80 --nt 81 --nx 80 --n_epochs 500000

# Reaction
# python -m src.experiments.pdes.simple.reaction

# Wave
# python -m src.experiments.pdes.simple.wave

# Burgers
# python -m src.experiments.pdes.benchmarks.burgers

# Allen-Cahn
# python -m src.experiments.pdes.benchmarks.allen_cahn

# NS 2D
# python -m src.experiments.pdes.benchmarks.ns_2d
