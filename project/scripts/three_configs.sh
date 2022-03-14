#!/bin/bash
#SBATCH --job-name=beaufils-first-run
#SBATCH --gres=gpu:4
#SBATCH --qos=qos_gpu-t4
#SBATCH --cpus-per-task=5
#SBATCH --output=./beaufils-first-run.out
#SBATCH --error=./beaufils-first-run.err
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
file_directory=feather
num_samples=50000
train_proportion=0.8
batch_size=64
goal_size=300
max_depth=10
num_epochs=5
reduction=both
learning_rate=1e-4
similarity=cosine
loss_function=cross_entropy
optimizer=sgd
total_floor=5
num_gpus=4
num_cpus=2
config=bow
srun --chdir ~/Thesis python3 ~/Thesis/run.py \
  --total_vocab \
  --folder_name $file_directory \
  --train_proportion $train_proportion \
  --indexes_size $num_samples \
  --batch_size $batch_size \
  --goal_size $goal_size \
  --max_depth $max_depth \
  --num_epochs $num_epochs \
  --lr $learning_rate \
  --similarity $similarity \
  --loss $loss_function \
  --optimizer $optimizer \
  --total_floor $total_floor \
  --num_gpus $num_gpus \
  --configuration $config \
  --num_cpus $num_cpus \
  --reduction $reduction

config=lstm

sbatch --chdir ~/Thesis python3 ~/Thesis/run.py \
  --total_vocab \
  --folder_name $file_directory \
  --train_proportion $train_proportion \
  --indexes_size $num_samples \
  --batch_size $batch_size \
  --goal_size $goal_size \
  --max_depth $max_depth \
  --num_epochs $num_epochs \
  --lr $learning_rate \
  --similarity $similarity \
  --loss $loss_function \
  --optimizer $optimizer \
  --total_floor $total_floor \
  --num_gpus $num_gpus \
  --configuration $config \
  --num_cpus $num_cpus

config=transformer
n_code=8
n_heads=8
shuffle=True
weight_decay=1e-4
drop_last=True
pin_memory=True
dropout=0.1
embedding_dim=128
reduction=both
dataloader=transformer

sbatch --chdir ~/Thesis python3 ~/Thesis/run.py \
  --total_vocab \
  --reduction $reduction \
  --folder_name $file_directory \
  --train_proportion $train_proportion \
  --indexes_size $num_samples \
  --batch_size $batch_size \
  --goal_size $goal_size \
  --max_depth $max_depth \
  --num_epochs $num_epochs \
  --lr $learning_rate \
  --similarity $similarity \
  --loss $loss_function \
  --optimizer $optimizer \
  --total_floor $total_floor \
  --num_gpus $num_gpus \
  --configuration $config \
  --num_cpus $num_cpus \
  --pin_memory $pin_memory \
  --drop_last $drop_last \
  --n_code $n_code \
  --n_heads $n_heads \
  --shuffle $shuffle \
  --weight_decay $weight_decay \
  --dataloader $dataloader \
  --dropout $dropout \
  --embedding_dim $embedding_dim
