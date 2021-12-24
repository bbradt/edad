#!/bin/sh
SITES=("2")
LR=("1e-4")
BATCH_SIZE=("32")
#DATASETS=("tapnet-NATOPS" "tapnet-Epilepsy" "tapnet-PEMS-SF" "tapnet-Heartbeat" "tapnet-SpokenArabicDigits")
DATASETS=("tapnet-CharacterTrajectories" "tapnet-BasicMotions" "tapnet-PenDigits" "tapnet-SpokenArabicDigits" "tapnet-NATOPS" "tapnet-PEMS-SF")
MODELS=("gru")
SPLITS=("class")
EXPNAME="gru_full"
RANK=("32")
EPOCHS=("100")
NUMITERATIONS=("10")
K=(0 1 2 3 4 5)
i=0
rank=8
numiter=10
for s in "${SITES[@]}"
do
    for lr in "${LR[@]}"
    do
        for epoch in "${EPOCHS[@]}"
        do
            for batch in "${BATCH_SIZE[@]}"
            do
                for data in "${DATASETS[@]}"
                do
                    for model in "${MODELS[@]}"
                    do
                        for split in "${SPLITS[@]}"
                        do                            
                            log_pref=${EXPNAME}_sites=${s}_lr=${lr}_batch=${batch}_data=${data}_split=${split}_epochs=${epoch}_rank=${rank}_numiter=${numiter}_
                            sbatch -J ${EXPNAME}${i} -o logs/${log_pref}.log -e logs/${log_pref}.err experiments/mode_runner.sh " --name ${EXPNAME} --n-sites ${s} --lr ${lr} --batch-size ${batch} --dataset ${data} --model ${model} --split ${split} --rank ${rank} --epochs ${epoch} --numiterations ${numiter}"
                        
                        done
                    done
                done
            done
        done
    done
done

