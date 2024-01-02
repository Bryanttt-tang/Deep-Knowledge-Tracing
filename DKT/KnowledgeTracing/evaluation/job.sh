#!/bin/bash

#SBATCH -n 2                             # Number of cores
#SBATCH --time=4-20:00                      # hours:minutes:seconds
#SBATCH --mem-per-cpu=80G
#SBATCH --tmp=4000                        # per node!!
#SBATCH --gpus=v100:1
#SBATCH --job-name=df1_embed_2lay_200
#SBATCH --output=./df1_embed_2lay_200.out
#SBATCH --error=./df1_embed_2lay_200.err

# run experiment
python ./run.py