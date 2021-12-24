#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task 4
#SBATCH -p qTRDGPUL

#SBATCH --gpus 1
#SBATCH -A PSYC0002
#SBATCH --oversubscribe
#SBATCH --exclude trendsagn011.rs.gsu.edu

eval "$(conda shell.bash hook)"
conda activate pytorch3
cd /data/users2/bbaker/projects/new_dad
KS=(0 1 2 3 4 5 6 7 8 9)
additional_args=$1
W=${2:-2}
echo $additional_args
i=0
for k in "${KS[@]}" 
do
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python dadnet/main.py --k $k $additional_args &
i=$((i + 1))
# Waiting
if [ $i -eq $W ]
then  
    for job in `jobs -p`
    do
        echo JOB $job
        wait $job || let "FAIL+=1"
    done
    echo DONE WAITING on $W
    i=0
fi
    # End Waiting block
done


