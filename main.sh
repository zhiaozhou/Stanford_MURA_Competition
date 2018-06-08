cd /home/paperspace/stanford/slim
nohup python train_image_classifier.py \
    --train_dir=/home/paperspace/stanford/train_logs \
    --dataset_dir=/home/paperspace/stanford/data_all_tfrecord/train \
    --num_samples=34534 \
    --num_classes=2 \
    --model_name=inception_resnet_v2 \
    --labels_to_names_path=/home/paperspace/stanford/data_all_tfrecord/labels.txt \
    --learning_rate=0.01 \
    --learning_rate_decay_factor=0.5\
    --num_epochs_per_decay=50 \
    --ignore_missing_vars=True \
    --max_number_of_steps=240000 \
    --optimizer=adam \
    --batch_size=64 > output.log 2>&1 & 
    
    
    
python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=flowers \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --model_name=nasnet_mobile    
    
CHECKPOINT_PATH=./pre-trained/nasnet-a_large_04_10_2017/model.ckpt
python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=flowers \
    --dataset_split_name=train \
    --batch_size=32 \
    --model_name=nasnet_large \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_exclude_scopes=final_layer,aux_11 \
    --trainable_scopes=final_layer,aux_11
    
    
nohup python train_image_classifier.py     --train_dir=/home/paperspace/stanford/train_logs     --dataset_dir=/home/paperspace/stanford/data_all_tfrecord/train     --num_samples=34534     --num_classes=2     --model_name=inception_resnet_v2     --labels_to_names_path=/home/paperspace/stanford/data_all_tfrecord/labels.txt   --checkpoint_path=/home/paperspace/stanford/checkpoints/inception_resnet_v2_2016_08_30.ckpt --checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits --trainable_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits  --learning_rate=1  --learning_rate_decay_factor=0.5    --num_epochs_per_decay=50     --ignore_missing_vars=True     --max_number_of_steps=240000     --optimizer=adam \

nohup python train_image_classifier.py     --train_dir=/home/paperspace/stanford/train_logs     --dataset_dir=/home/paperspace/stanford/data_all_tfrecord/train     --num_samples=34534     --num_classes=2     --model_name=inception_resnet_v2     --labels_to_names_path=/home/paperspace/stanford/data_all_tfrecord/labels.txt  --learning_rate=1  --learning_rate_decay_factor=2.3  --num_epochs_per_decay=1000     --max_number_of_steps=240000 --batch_size=48  --optimizer=adadelta > output.log 2>&1 & 

python eval_image_classifier.py \
    --checkpoint_path=/home/paperspace/stanford/train_logs  \
    --eval_dir=/home/paperspace/stanford/eval_logs \
    --dataset_dir=/home/paperspace/stanford/data_all_tfrecord/validation \
    --num_samples=2778 \
    --num_classes=2 \
    --model_name=inception_resnet_v2