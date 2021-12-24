#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task 10
#SBATCH -p qTRDGPU

#SBATCH --gpus 1
#SBATCH -A PSYC0002
#SBATCH --oversubscribe
#SBATCH --exclude trendsagn011.rs.gsu.edu

eval "$(conda shell.bash hook)"
conda activate pytorch3
cd /data/users2/bbaker/projects/new_dad
#KS=(0 1 2 3 4 5 6 7 8 9)
additional_args=$1
k=$2
echo $additional_args
i=0
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python dadnet/main.py --k $k $additional_args
i=$((i + 1))


