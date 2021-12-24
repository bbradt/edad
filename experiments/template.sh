#!/bin/sh
SITES=("2")
LR=("1e-6")
BATCH_SIZE=("32")
DATASETS=("mnist")
MODELS=("simpleff")
MODES=("pooled" "dsgd" "dad" "edad" "rankdad")
SPLITS=("class")
EXPNAME="test"
RANK=("32")
EPOCHS=("200")
NUMITERATIONS=("10")
K=(0 1 2 3 4 5 6 7 8 9 10)
i=0
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
                            for mode in "${MODES[@]}"
                            do
                                for rank in "${RANK[@]}"
                                do
                                    for numiter in "${NUMITERATIONS[@]}"
                                    do
                                        for k in "${K[@}]}"
                                            log_pref=${EXPNAME}_sites=${s}_lr=${lr}_batch=${batch}_data=${data}_split=${split}_mode=${mode}_rank=${rank}_numiter=${numiter}_epochs=${epoch}_k=${k}
                                            sbatch -J ${EXPNAME}${i} -o logs/${log_pref}.log -e logs/${log_pref}.err experiments/runner.sh " --name ${EXPNAME} --n-sites ${s} --lr ${lr} --batch-size ${batch} --dataset ${data} --model ${model} --split ${split} --mode ${mode} --rank ${rank} --k ${k} --epochs ${epoch} --numiterations ${numiter}"
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

