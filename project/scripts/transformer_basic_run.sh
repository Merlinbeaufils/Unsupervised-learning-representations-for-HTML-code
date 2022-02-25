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
home_directory=~/Thesis
file_directory=common_sites
num_samples=5000
train_proportion=0.8
batch_size=64
goal_size=300
max_depth=40
num_epochs=5
learning_rate=2e-3
weight_decay=1e-4
dataloader=transformer
similarity=cosine
loss_function=cross_entropy
optimizer=sgd
total_floor=0
num_gpus=4
num_cpus=2
config=transformer
n_code=8
n_heads=8
shuffle=True
drop_last=True
pin_memory=True
dropout=0.1
embedding_dim=128
reduction=both

srun --chdir $home_directory python3 ~/Thesis/run.py \
  --build_trees \
  --build_vocabs \
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




