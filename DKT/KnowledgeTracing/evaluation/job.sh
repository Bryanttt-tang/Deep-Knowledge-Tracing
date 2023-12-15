#!/bin/bash

#SBATCH -n 2                             # Number of cores
#SBATCH --time=5-20:00                      # hours:minutes:seconds
#SBATCH --mem-per-cpu=40G
#SBATCH --tmp=4000                        # per node!!
#SBATCH --gpus=rtx_3090:1
#SBATCH --job-name=df1_embed
#SBATCH --output=./df1_embed.out
#SBATCH --error=./df1_embed.err

# run experiment
python ./run.py