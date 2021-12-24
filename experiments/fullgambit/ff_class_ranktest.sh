#!/bin/sh
SITES=("2")
LR=("1e-4")
BATCH_SIZE=("32")
DATASETS=("mnist")
MODELS=("simpleff")
SPLITS=("class")
EXPNAME="gru_full"
RANK=("32" "16" "8" "4" "3")
EPOCHS=("50")
NUMITERATIONS=("10")
K=(0 1 2 3 4 5)
i=0
rank=32
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
                            for rank in "${RANK[@]}"
                            do                                                                
                                    log_pref=_ff${EXPNAME}_sites=${s}_lr=${lr}_batch=${batch}_data=${data}_split=${split}_epochs=${epoch}_rank=${rank}_numiter=${numiter}_
                                    sbatch -J ${EXPNAME}${i} -o logs/${log_pref}.log -e logs/${log_pref}.err experiments/rank_mode_runner.sh " --name ${EXPNAME} --n-sites ${s} --lr ${lr} --batch-size ${batch} --dataset ${data} --model ${model} --split ${split} --rank ${rank} --epochs ${epoch} --numiterations ${numiter}"                                
                            done                                                            
                        done
                    done
                done
            done
        done
    done
done

