#!/bin/sh
SITES=("2")
LR=("1e-4")
BATCH_SIZE=("32")
DATASETS=("imdb")
MODELS=("transformer")
MODES=("dsgd" "dad" "noshare" "rankdad")
SPLITS=("random")
EXPNAME="transformer"
RANK=("8")
EPOCHS=("50")
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
                for data in "${DATASETS[@]}"
                do
                    for model in "${MODELS[@]}"
                    do
                        for split in "${SPLITS[@]}"
                        do          
                        for k in "${K[@]}"               
                        do
                        for rank in "${RANK[@]}"
                        do
                        log_pref=${EXPNAME}_sites=${s}_lr=${lr}_batch=${batch}_data=${data}_split=${split}_epochs=${epoch}_rank=${rank}_numiter=${numiter}_k=${k}
                        sbatch -J pooledk$k${EXPNAME}${i} -o logs/${log_pref}.log -e logs/${log_pref}.err experiments/transformer_runner.sh " --name ${EXPNAME} --n-sites ${s} --lr ${lr} --dataset ${data} --model ${model} --split ${split} --rank ${rank} --epochs ${epoch} --numiterations ${numiter} --k ${k} --mode pooled --batch-size 32"
                        for mode in "${MODES[@]}"
                        do
                            log_pref=${EXPNAME}_sites=${s}_lr=${lr}_batch=${batch}_data=${data}_split=${split}_epochs=${epoch}_rank=${rank}_numiter=${numiter}_k=${k}_mode=${mode}
                            sbatch -J ${mode}k$k${EXPNAME}${i} -o logs/${log_pref}.log -e logs/${log_pref}.err experiments/transformer_runner.sh " --name ${EXPNAME} --n-sites ${s} --lr ${lr} --dataset ${data} --model ${model} --split ${split} --rank ${rank} --epochs ${epoch} --numiterations ${numiter} --k ${k} --mode ${mode} --batch-size 16"
                        done
                        done                        
                        done
                        done
                    done
                done            
        done
    done
done

