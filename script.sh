#!/bin/bash
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --time=2-0
#SBATCH --array=0-15

ml python/3.6.1
ml py-tensorflow/2.4.1_py36
ml cuda/11.5.0
ml cudnn/8.1.1.33

python3 inference-char.py $SLURM_ARRAY_TASK_ID

### sbatch -o sbatch.out script.sh
### squeue --me
### scancel -u akashl
