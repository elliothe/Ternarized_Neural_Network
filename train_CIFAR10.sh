#!/usr/bin/env sh

PYTHON="/home/elliot/anaconda3/envs/pytorch_041/bin/python"

############ directory to save result #############
DATE=`date +%Y-%m-%d`

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
    mkdir ./dataset
    mkdir ./save/${DATE}/
fi

############ Configurations ###############
model=tern_resnet20
dataset=cifar10
epochs=200
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=test

$PYTHON main.py --dataset ${dataset} \
    --data_path ./dataset/   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120 160  --gammas 0.1 0.1 0.5 \
    --batch_size ${batch_size} --workers 4 --ngpu 1  \
    --print_freq 100 --decay 0.0003 \
    #--resume ${pretrained_model} --evaluate\
    #--model_only  --fine_tune\
  
