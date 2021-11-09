#!/bin/bash
#SBATCH -p gpu
#SBATCH -c 10
#SBATCH -G 1

ml python/3.6.1
ml py-tensorflow/2.4.1_py36
ml cuda/11.5.0
ml cudnn/8.1.1.33


