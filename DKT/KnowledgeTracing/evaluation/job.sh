#!/bin/bash

SBATCH -n 16                              # Number of cores
SBATCH --time=1:00:00                      # hours:minutes:seconds
SBATCH --mem-per-cpu=2000
SBATCH --tmp=4000                        # per node!!
SBATCH --job-name=simple_job
SBATCH --output=./simple_job.out
SBATCH --error=./simple_job.err

# run experiment
python ./run.py