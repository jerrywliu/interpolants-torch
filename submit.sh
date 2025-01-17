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
# python -m src.experiments.pdes.simple.advection
# python -m src.experiments.pdes.simple.reaction
# python -m src.experiments.pdes.simple.wave
python -m src.experiments.pdes.benchmarks.burgers
# python -m src.experiments.pdes.benchmarks.allen_cahn
# python -m src.experiments.pdes.benchmarks.ns_2d