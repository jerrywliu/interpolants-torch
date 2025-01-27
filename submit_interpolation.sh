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

# Interpolation
# Advection
# python -m src.experiments.pdes.simple.advection --c 40 --n_t 81 --n_x 80 --n_epochs 5000000 --method adam
# python -m src.experiments.pdes.simple.advection --c 40 --n_t 81 --n_x 80 --n_epochs 100000 --method nys_newton --sample_type standard
# python -m src.experiments.pdes.simple.advection --c 40 --sample_type uniform --method adam --n_epochs 1000000 --model mlp

# Reaction
python -m src.experiments.interpolation.pde_solns.reaction --rho 5 --n_t 41 --n_x 41 --n_epochs 10000 --method adam --model rational
# python -m src.experiments.pdes.simple.reaction --rho 5 --n_t 41 --n_x 41 --n_epochs 5000000 --method adam
# python -m src.experiments.pdes.simple.reaction --rho 5 --n_t 41 --n_x 41 --n_epochs 100000 --method nys_newton --sample_type standard
# python -m src.experiments.pdes.simple.reaction --rho 5 --sample_type uniform --method adam --n_epochs 1000000 --model mlp

# Wave
# python -m src.experiments.pdes.simple.wave --c 2 --beta 5 --n_t 41 --n_x 41 --n_epochs 100000 --method nys_newton --sample_type standard --model polynomial
# python -m src.experiments.pdes.simple.wave --c 2 --beta 5 --n_t 41 --n_x 41 --n_epochs 5000000 --method adam --sample_type standard --model polynomial
# python -m src.experiments.pdes.simple.wave --c 2 --beta 5 --sample_type uniform --method adam --n_epochs 1000000 --model mlp
# python -m src.experiments.pdes.simple.wave --c 2 --beta 5 --n_t 41 --n_x 41 --n_epochs 100000 --eval_every 100 --method nys_newton --sample_type standard --model polynomial_noisy
# python -m src.experiments.pdes.simple.wave --c 2 --beta 5 --n_t 41 --n_x 41 --n_epochs 100000 --eval_every 1000 --method adam --sample_type standard --model polynomial_noisy

# Burgers
# python -m src.experiments.pdes.benchmarks.burgers --n_t 81 --n_x 81 --n_epochs 100000 --method nys_newton --sample_type standard
# python -m src.experiments.pdes.benchmarks.burgers --n_t 81 --n_x 81 --n_epochs 100000 --eval_every 100 --method nys_newton --sample_type standard --model polynomial

# Allen-Cahn
# python -m src.experiments.pdes.benchmarks.allen_cahn --n_t 81 --n_x 81 --n_epochs 100000 --method nys_newton --sample_type standard
# python -m src.experiments.pdes.benchmarks.allen_cahn --n_t 81 --n_x 81 --n_epochs 100000 --eval_every 100 --method nys_newton --sample_type standard --model polynomial
