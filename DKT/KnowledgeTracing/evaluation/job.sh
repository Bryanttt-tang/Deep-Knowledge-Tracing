#!/bin/bash

#SBATCH -n 2                             # Number of cores
#SBATCH --time=4-20:00                      # hours:minutes:seconds
#SBATCH --mem-per-cpu=80G
#SBATCH --tmp=4000                        # per node!!
#SBATCH --gpus=rtx_3090:1
#SBATCH --job-name=c4_lgr_added
#SBATCH --output=./c4_lgr_added.out
#SBATCH --error=./c4_lgr_added.err

# run experiment
python ./run.py