#!/bin/bash

# 1 GPU, 4 CPUs, 2 hours
#SBATCH -p gpu --gres=gpu:1
#SBATCH -n 4
#SBATCH -t 02:00:00

#SBATCH -J spinsim
#SBATCH -o echojob-%j.out
#SBATCH -e echojob-%j.err

module load julia/1.5.2
module load cuda

julia sample_sim.jl



