#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task 10
#SBATCH -p qTRDGPU

#SBATCH --gpus 1
#SBATCH -A PSYC0002
#SBATCH --oversubscribe
#SBATCH -J DDR-LW
#SBATCH -o slurm/logs/ddr-lw.out
#SBATCH -e slurm/logs/ddr-lw.err
#SBATCH --exclude trendsagn011.rs.gsu.edu

eval "$(conda shell.bash hook)"
conda activate pytorch3
cd /data/users2/bbaker/projects/new_dad

MODES=("pooled" "edad" "dsgd" "noshare")
SPLITS=("class" "random")
additional_args=$1
echo $additional_args
for split in "${SPLITS[@]}"
do
for mode in "${MODES[@]}"
do
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python dadnet/main.py --mode $mode --split $split $additional_args &
done
done
for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done

