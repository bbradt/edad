#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task 8
#SBATCH -p qTRDGPUL

#SBATCH --gpus 1
#SBATCH -A PSYC0002
#SBATCH --oversubscribe
#SBATCH --exclude trendsdgxa001.rs.gsu.edu

eval "$(conda shell.bash hook)"
conda activate pytorch3
cd /data/users2/bbaker/projects/new_dad
#KS=(0 1 2 3 4)
#MODES=("dad" "rankdad" "noshare" "dsgd")
additional_args=$1
echo $additional_args
i=1
W=1
#for k in "${KS[@]}"
#do
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python dadnet/main.py $additional_args
#for mode in "${MODES[@]}"
#do
#CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python dadnet/main.py $additional_args --mode $mode --batch-size 16&\
#i=$((i + 1))
#if [ $i -ge $W ]
#    then  
#    for job in `jobs -p`
#    do
#        echo JOB $job
#        wait $job || let "FAIL+=1"
#    done
#    echo DONE WAITING on $W
#    i=0
#fi
#done
#done

