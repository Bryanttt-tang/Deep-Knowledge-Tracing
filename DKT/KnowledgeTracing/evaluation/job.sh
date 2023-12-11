#!/bin/bash

#SBATCH -n 2                             # Number of cores
#SBATCH --time=3-20:00                      # hours:minutes:seconds
#SBATCH --mem-per-cpu=40G
#SBATCH --tmp=4000                        # per node!!
#SBATCH --gpus=rtx_3090:1
#SBATCH --job-name=ASS17_baseline
#SBATCH --output=./ASS17_baseline.out
#SBATCH --error=./ASS17_baseline.err

# run experiment
python ./run.py