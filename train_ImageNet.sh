#!/usr/bin/env sh

PYTHON="/home/elliot/anaconda3/envs/pytorch_041/bin/python"
imagenet_path="/media/elliot/20744C7E744C58A4/Users/Elliot_he/Documents/imagenet"
pretrained_model=/home/elliot/Documents/WACV_2019/code_github/ResNet_imagenet/imagenet_quan_resnet18b_fq_lq_50_expansion_4/model_best.pth.tar

############ directory to save result #############
DATE=`date +%Y-%m-%d`

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
    mkdir ./save/${DATE}/
fi

############ Configurations ###############
model=resnet18b_fq_lq_tern_tex_4
dataset=imagenet
epochs=50
batch_size=256
optimizer=Adam
# add more labels as additional info into the saving path
label_info=test2

$PYTHON main.py --dataset ${dataset} \
    --data_path ${imagenet_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${label_info} \
    --epochs ${epochs} --learning_rate 0.0001 \
    --optimizer ${optimizer} \
	--schedule 30 40 45  --gammas 0.2 0.2 0.5 \
    --batch_size ${batch_size} --workers 8 --ngpu 2  \
    --print_freq 100 --decay 0.000005 \
    --resume ${pretrained_model} --evaluate\
    --model_only  --fine_tune\
  
