#!/bin/bash

#SBATCH -n 2                             # Number of cores
#SBATCH --time=4-20:00                      # hours:minutes:seconds
#SBATCH --mem-per-cpu=40G
#SBATCH --tmp=4000                        # per node!!
#SBATCH --gpus=rtx_3090:2
#SBATCH --job-name=df1_=250
#SBATCH --output=./df1_=250.out
#SBATCH --error=./df1_=250.err

# run experiment
python ./run.py