#!/bin/bash

export PYTHONPATH=.


python  ./run.py \
--folder_name final_feather \
--num_trees 7500 \
--index_config per_tree \
--indexes_per_tree 5 \
--index_config per_tree \
--max_tree_size 1000 \
--max_depth 10 \
--total_floor 50 \
--reduction both \
--sample_config transformer \
--indexes_size 40000 \
--stop


python  ./run.py \
--folder_name final_feather \
--num_trees 1700 \
--index_config per_tree \
--indexes_per_tree 5 \
--index_config per_tree \
--max_tree_size 1000 \
--max_depth 10 \
--total_floor 50 \
--reduction both \
--sample_config transformer \
--indexes_size 1700 \
--stop



python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration bow \
--num_epochs 50 \
--lr 1e-4 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name large_run2

python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration lstm \
--num_epochs 50 \
--lr 1 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name large_run2

python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration transformer \
--num_epochs 50 \
--lr 1e-2 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name large_run2

python  ./run.py \
--folder_name final_feather \
--framework finetune \
--skip_setup \
--configuration bow \
--num_epochs 50 \
--lr 1e-4 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name large_run2

python  ./run.py \
--folder_name final_feather \
--framework finetune \
--skip_setup \
--configuration lstm \
--num_epochs 50 \
--lr 1 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name large_run2

python  ./run.py \
--folder_name final_feather \
--framework finetune \
--skip_setup \
--configuration transformer \
--num_epochs 50 \
--lr 1e-2 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name large_run2

python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration lstm \
--num_epochs 50 \
--lr 5e-1 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name large_small_lr

python  ./run.py \
--folder_name final_feather \
--framework finetune \
--skip_setup \
--configuration lstm \
--num_epochs 50 \
--lr 5e-1 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name large_small_lr

python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration lstm \
--num_epochs 50 \
--lr 5 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name large_large_lr

python  ./run.py \
--folder_name final_feather \
--framework finetune \
--skip_setup \
--configuration lstm \
--num_epochs 50 \
--lr 5 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name large_large_lr





python  ./run.py \
--folder_name final_feather \
--num_trees 7500 \
--index_config per_tree \
--indexes_per_tree 5 \
--index_config per_tree \
--max_tree_size 200 \
--max_depth 10 \
--total_floor 40 \
--reduction both \
--sample_config transformer \
--indexes_size 40000 \
--no_keys \
--stop


python  ./run.py \
--folder_name final_feather \
--num_trees 1700 \
--index_config per_tree \
--indexes_per_tree 5 \
--index_config per_tree \
--max_tree_size 1000 \
--max_depth 10 \
--total_floor 50 \
--reduction both \
--sample_config transformer \
--indexes_size 1700 \
--no_keys \
--stop

python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration bow \
--num_epochs 50 \
--lr 1e-4 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name large_edited_bow

python  ./run.py \
--folder_name final_feather \
--framework finetune \
--skip_setup \
--configuration bow \
--num_epochs 50 \
--lr 1e-4 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name large_edited_bow




python  ./run.py \
--folder_name final_feather \
--num_trees 1000 \
--index_config per_tree \
--indexes_per_tree 5 \
--index_config per_tree \
--max_tree_size 1000 \
--max_depth 10 \
--total_floor 100 \
--reduction both \
--sample_config transformer \
--indexes_size 5000 \
--stop

python  ./run.py \
--folder_name final_feather \
--num_trees 1700 \
--framework finetune \
--index_config per_tree \
--index_config per_tree \
--max_tree_size 1000 \
--max_depth 10 \
--total_floor 100 \
--reduction both \
--sample_config transformer \
--indexes_size 1700 \
--stop


python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration bow \
--num_epochs 10 \
--lr 1e-4 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name embedding_32

python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration lstm \
--num_epochs 10 \
--lr 1 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name embedding_32

python  ./run.py \
--folder_name final_feather \
--framework finetune \
--skip_setup \
--configuration bow \
--num_epochs 10 \
--lr 1e-4 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name embedding_32

python  ./run.py \
--folder_name final_feather \
--framework finetune \
--skip_setup \
--configuration lstm \
--num_epochs 10 \
--lr 1 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name embedding_32



python  ./run.py \
--folder_name final_feather \
--num_trees 1000 \
--index_config per_tree \
--indexes_per_tree 5 \
--index_config per_tree \
--max_tree_size 1000 \
--max_depth 10 \
--total_floor 10 \
--reduction both \
--sample_config transformer \
--indexes_size 5000 \
--stop

python  ./run.py \
--folder_name final_feather \
--num_trees 1700 \
--framework finetune \
--index_config per_tree \
--index_config per_tree \
--max_tree_size 1000 \
--max_depth 10 \
--total_floor 10 \
--reduction both \
--sample_config transformer \
--indexes_size 1700 \
--stop


python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration bow \
--num_epochs 10 \
--lr 1e-4 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 64 \
--experiment_name embedding_64

python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration lstm \
--num_epochs 10 \
--lr 1 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 64 \
--experiment_name embedding_64

python  ./run.py \
--folder_name final_feather \
--framework finetune \
--skip_setup \
--configuration bow \
--num_epochs 10 \
--lr 1e-4 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 64 \
--experiment_name embedding_64

python  ./run.py \
--folder_name final_feather \
--framework finetune \
--skip_setup \
--configuration lstm \
--num_epochs 10 \
--lr 1 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 64 \
--experiment_name embedding_64


python  ./run.py \
--folder_name final_feather \
--num_trees 1000 \
--index_config per_tree \
--indexes_per_tree 5 \
--index_config per_tree \
--max_tree_size 1000 \
--max_depth 10 \
--total_floor 50 \
--reduction both \
--sample_config transformer \
--indexes_size 5000 \
--stop

python  ./run.py \
--folder_name final_feather \
--num_trees 1700 \
--framework finetune \
--index_config per_tree \
--max_tree_size 1000 \
--max_depth 10 \
--total_floor 50 \
--reduction both \
--sample_config transformer \
--indexes_size 1700 \
--stop


python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration bow \
--num_epochs 10 \
--lr 1e-4 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 128 \
--experiment_name embedding_128

python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration lstm \
--num_epochs 10 \
--lr 1 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 128 \
--experiment_name embedding_128

python  ./run.py \
--folder_name final_feather \
--framework finetune \
--skip_setup \
--configuration bow \
--num_epochs 10 \
--lr 1e-4 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 128 \
--experiment_name embedding_128

python  ./run.py \
--folder_name final_feather \
--framework finetune \
--skip_setup \
--configuration lstm \
--num_epochs 10 \
--lr 1 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 128 \
--experiment_name embedding_128







