#!/bin/bash

#SBATCH -n 2                             # Number of cores
#SBATCH --time=3-20:00                      # hours:minutes:seconds
#SBATCH --mem-per-cpu=40G
#SBATCH --tmp=4000                        # per node!!
#SBATCH --gpus=v100:1
#SBATCH --job-name=df1_LSTM
#SBATCH --output=./df1_LSTM.out
#SBATCH --error=./df1_LSTM.err

# run experiment
python ./run.py