#!/bin/bash

#SBATCH -n 2                             # Number of cores
#SBATCH --time=3-20:00                      # hours:minutes:seconds
#SBATCH --mem-per-cpu=40G
#SBATCH --tmp=4000                        # per node!!
#SBATCH --gpus=rtx_3090:1
#SBATCH --job-name=sem1_lgr
#SBATCH --output=./sem1_lgr.out
#SBATCH --error=./sem1_lgr.err

# run experiment
python ./run.py