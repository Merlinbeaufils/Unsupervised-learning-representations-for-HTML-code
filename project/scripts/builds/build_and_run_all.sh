#!/bin/bash

export PYTHONPATH=.


python  ./run.py \
--folder_name final_feather \
--num_trees 100 \
--index_config per_tree \
--indexes_per_tree 5 \
--index_config per_tree \
--max_tree_size 500 \
--max_depth 10 \
--total_floor 100 \
--reduction both \
--sample_config transformer \
--indexes_size 500 \
--stop

python  ./run.py \
--folder_name final_feather \
--num_trees 100 \
--framework finetune \
--index_config per_tree \
--max_tree_size 500 \
--max_depth 10 \
--total_floor 100 \
--reduction both \
--sample_config transformer \
--indexes_size 100 \
--stop



python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration bow \
--num_epochs 1 \
--lr 1e-4 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name ignore

python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration lstm \
--num_epochs 1 \
--lr 1 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name ignore

python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration transformer \
--num_epochs 1 \
--lr 1e-2 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name ignore

python  ./run.py \
--folder_name final_feather \
--framework finetune \
--skip_setup \
--configuration bow \
--num_epochs 1 \
--lr 1e-4 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name ignore

python  ./run.py \
--folder_name final_feather \
--framework finetune \
--skip_setup \
--configuration lstm \
--num_epochs 1 \
--lr 1 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name ignore

python  ./run.py \
--folder_name final_feather \
--framework finetune \
--skip_setup \
--configuration transformer \
--num_epochs 1 \
--lr 1e-2 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name ignore






python  ./run.py \
--folder_name final_feather \
--num_trees 100 \
--index_config per_tree \
--indexes_per_tree 5 \
--index_config per_tree \
--max_tree_size 500 \
--max_depth 10 \
--total_floor 100 \
--reduction both \
--sample_config transformer \
--indexes_size 500 \
--stop

python  ./run.py \
--folder_name final_feather \
--num_trees 100 \
--framework finetune \
--index_config per_tree \
--indexes_per_tree 5 \
--max_tree_size 500 \
--max_depth 10 \
--total_floor 100 \
--reduction both \
--sample_config transformer \
--indexes_size 500 \
--stop

python  ./run.py \
--folder_name final_feather \
--dont_build_trees \
--index_config per_tree \
--indexes_per_tree 5 \
--total_floor 100 \
--reduction both \
--sample_config transformer \
--indexes_size 500 \
--data_config only_depth \
--stop

python  ./run.py \
--folder_name final_feather \
--framework finetune \
--dont_build_trees \
--index_config per_tree \
--total_floor 100 \
--reduction both \
--sample_config transformer \
--indexes_size 500 \
--data_config only_depth \
--stop

python  ./run.py \
--folder_name final_feather \
--dont_build_trees \
--index_config per_tree \
--indexes_per_tree 5 \
--index_config per_tree \
--total_floor 100 \
--reduction both \
--sample_config transformer \
--indexes_size 500 \
--data_config only_tag \
--stop

python  ./run.py \
--folder_name final_feather \
--framework finetune \
--dont_build_trees \
--index_config per_tree \
--total_floor 100 \
--reduction both \
--sample_config transformer \
--indexes_size 500 \
--data_config only_tag \
--stop

python  ./run.py \
--folder_name final_feather \
--dont_build_trees \
--index_config per_tree \
--indexes_per_tree 10 \
--index_config per_tree \
--total_floor 100 \
--reduction both \
--sample_config transformer \
--indexes_size 500 \
--data_config no_depth \
--stop

python  ./run.py \
--folder_name final_feather \
--framework finetune \
--dont_build_trees \
--index_config per_tree \
--total_floor 100 \
--reduction both \
--sample_config transformer \
--indexes_size 500 \
--data_config no_depth \
--stop

python  ./run.py \
--folder_name final_feather \
--dont_build_trees \
--index_config per_tree \
--indexes_per_tree 5 \
--index_config per_tree \
--total_floor 100 \
--reduction both \
--sample_config transformer \
--indexes_size 500 \
--data_config no_keys \
--stop

python  ./run.py \
--folder_name final_feather \
--framework finetune \
--dont_build_trees \
--index_config per_tree \
--total_floor 100 \
--reduction both \
--sample_config transformer \
--indexes_size 500 \
--data_config no_keys \
--stop

python  ./run.py \
--folder_name final_feather \
--dont_build_trees \
--index_config per_tree \
--indexes_per_tree 5 \
--total_floor 100 \
--reduction both \
--sample_config transformer \
--indexes_size 500 \
--data_config key_only \
--stop

python  ./run.py \
--folder_name final_feather \
--framework finetune \
--dont_build_trees \
--index_config per_tree \
--total_floor 100 \
--reduction both \
--sample_config transformer \
--indexes_size 500 \
--data_config key_only \
--stop




python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration bow \
--num_epochs 1 \
--lr 1e-4 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name ignore

python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration lstm \
--num_epochs 1 \
--lr 1 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name ignore

python  ./run.py \
--folder_name final_feather \
--framework finetune \
--skip_setup \
--configuration bow \
--num_epochs 1 \
--lr 1e-4 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name ignore

python  ./run.py \
--folder_name final_feather \
--framework finetune \
--skip_setup \
--configuration lstm \
--num_epochs 1 \
--lr 1 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name ignore


python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration bow \
--num_epochs 1 \
--lr 1e-4 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--data_config no_keys \
--experiment_name ignore

python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration lstm \
--num_epochs 1 \
--lr 1 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--data_config no_keys \
--experiment_name ignore

python  ./run.py \
--folder_name final_feather \
--framework finetune \
--skip_setup \
--configuration bow \
--num_epochs 1 \
--lr 1e-4 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--data_config no_keys \
--experiment_name ignore

python  ./run.py \
--folder_name final_feather \
--framework finetune \
--skip_setup \
--configuration lstm \
--num_epochs 1 \
--lr 1 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--data_config no_keys \
--experiment_name ignore


python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration bow \
--num_epochs 1 \
--lr 1e-4 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--data_config no_depth \
--experiment_name ignore

python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration lstm \
--num_epochs 1 \
--lr 1 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--data_config no_depth \
--experiment_name ignore

python  ./run.py \
--folder_name final_feather \
--framework finetune \
--skip_setup \
--configuration bow \
--num_epochs 1 \
--lr 1e-4 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--data_config no_depth \
--experiment_name ignore

python  ./run.py \
--folder_name final_feather \
--framework finetune \
--skip_setup \
--configuration lstm \
--num_epochs 1 \
--lr 1 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--data_config no_depth \
--experiment_name ignore


python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration bow \
--num_epochs 1 \
--lr 1e-4 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--data_config only_tag \
--experiment_name ignore

python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration lstm \
--num_epochs 1 \
--lr 1 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--data_config only_tag \
--experiment_name ignore

python  ./run.py \
--folder_name final_feather \
--framework finetune \
--skip_setup \
--configuration bow \
--num_epochs 1 \
--lr 1e-4 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--data_config only_tag \
--experiment_name ignore

python  ./run.py \
--folder_name final_feather \
--framework finetune \
--skip_setup \
--configuration lstm \
--num_epochs 1 \
--lr 1 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--data_config only_tag \
--experiment_name ignore


python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration bow \
--num_epochs 1 \
--lr 1e-4 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--data_config only_depth \
--experiment_name ignore

python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration lstm \
--num_epochs 1 \
--lr 1 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--data_config only_depth \
--experiment_name ignore

python  ./run.py \
--folder_name final_feather \
--framework finetune \
--skip_setup \
--configuration bow \
--num_epochs 1 \
--lr 1e-4 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--data_config only_depth \
--experiment_name ignore

python  ./run.py \
--folder_name final_feather \
--framework finetune \
--skip_setup \
--configuration lstm \
--num_epochs 1 \
--lr 1 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--data_config only_depth \
--experiment_name ignore


python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration bow \
--num_epochs 1 \
--lr 1e-4 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--data_config key_only \
--experiment_name ignore

python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration lstm \
--num_epochs 1 \
--lr 1 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--data_config key_only \
--experiment_name ignore

python  ./run.py \
--folder_name final_feather \
--framework finetune \
--skip_setup \
--configuration bow \
--num_epochs 1 \
--lr 1e-4 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--data_config key_only \
--experiment_name ignore

python  ./run.py \
--folder_name final_feather \
--framework finetune \
--skip_setup \
--configuration lstm \
--num_epochs 1 \
--lr 1 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--data_config key_only \
--experiment_name ignore



python  ./run.py \
--folder_name final_feather \
--num_trees 100 \
--index_config per_tree \
--indexes_per_tree 5 \
--index_config per_tree \
--max_tree_size 1000 \
--max_depth 10 \
--total_floor 100 \
--reduction both \
--sample_config transformer \
--indexes_size 500 \
--stop

python  ./run.py \
--folder_name final_feather \
--num_trees 100 \
--framework finetune \
--index_config per_tree \
--max_tree_size 1000 \
--max_depth 10 \
--total_floor 100 \
--reduction both \
--sample_config transformer \
--indexes_size 500 \
--stop


python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration bow \
--num_epochs 1 \
--lr 1e-4 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name ignore

python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration lstm \
--num_epochs 1 \
--lr 1 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name ignore

python  ./run.py \
--folder_name final_feather \
--framework finetune \
--skip_setup \
--configuration bow \
--num_epochs 1 \
--lr 1e-4 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name ignore

python  ./run.py \
--folder_name final_feather \
--framework finetune \
--skip_setup \
--configuration lstm \
--num_epochs 1 \
--lr 1 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name ignore



python  ./run.py \
--folder_name final_feather \
--num_trees 100 \
--index_config per_tree \
--indexes_per_tree 5 \
--index_config per_tree \
--max_tree_size 500 \
--max_depth 10 \
--total_floor 100 \
--reduction both \
--sample_config transformer \
--indexes_size 500 \
--stop

python  ./run.py \
--folder_name final_feather \
--num_trees 100 \
--framework finetune \
--index_config per_tree \
--max_tree_size 500 \
--max_depth 10 \
--total_floor 100 \
--reduction both \
--sample_config transformer \
--indexes_size 500 \
--stop


python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration bow \
--num_epochs 1 \
--lr 1e-4 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name ignore

python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration lstm \
--num_epochs 1 \
--lr 1 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name ignore

python  ./run.py \
--folder_name final_feather \
--framework finetune \
--skip_setup \
--configuration bow \
--num_epochs 1 \
--lr 1e-4 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name ignore

python  ./run.py \
--folder_name final_feather \
--framework finetune \
--skip_setup \
--configuration lstm \
--num_epochs 1 \
--lr 1 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name ignore


python  ./run.py \
--folder_name final_feather \
--num_trees 100 \
--index_config per_tree \
--indexes_per_tree 5 \
--index_config per_tree \
--max_tree_size 200 \
--max_depth 10 \
--total_floor 100 \
--reduction both \
--sample_config transformer \
--indexes_size 640 \
--stop

python  ./run.py \
--folder_name final_feather \
--num_trees 100 \
--framework finetune \
--index_config per_tree \
--max_tree_size 200 \
--max_depth 10 \
--total_floor 100 \
--reduction both \
--sample_config transformer \
--indexes_size 500 \
--stop


python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration bow \
--num_epochs 1 \
--lr 1e-4 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name ignore

python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration lstm \
--num_epochs 1 \
--lr 1 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name ignore

python  ./run.py \
--folder_name final_feather \
--framework finetune \
--skip_setup \
--configuration bow \
--num_epochs 1 \
--lr 1e-4 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name ignore

python  ./run.py \
--folder_name final_feather \
--framework finetune \
--skip_setup \
--configuration lstm \
--num_epochs 1 \
--lr 1 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name ignore





python  ./run.py \
--folder_name final_feather \
--num_trees 100 \
--index_config per_tree \
--indexes_per_tree 5 \
--index_config per_tree \
--max_tree_size 1000 \
--max_depth 10 \
--total_floor 100 \
--reduction both \
--sample_config transformer \
--indexes_size 500 \
--stop

python  ./run.py \
--folder_name final_feather \
--num_trees 100 \
--framework finetune \
--index_config per_tree \
--index_config per_tree \
--max_tree_size 1000 \
--max_depth 10 \
--total_floor 100 \
--reduction both \
--sample_config transformer \
--indexes_size 100 \
--stop


python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration bow \
--num_epochs 1 \
--lr 1e-4 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name ignore

python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration lstm \
--num_epochs 1 \
--lr 1 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name ignore

python  ./run.py \
--folder_name final_feather \
--framework finetune \
--skip_setup \
--configuration bow \
--num_epochs 1 \
--lr 1e-4 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name ignore

python  ./run.py \
--folder_name final_feather \
--framework finetune \
--skip_setup \
--configuration lstm \
--num_epochs 1 \
--lr 1 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name ignore



python  ./run.py \
--folder_name final_feather \
--num_trees 100 \
--index_config per_tree \
--indexes_per_tree 5 \
--index_config per_tree \
--max_tree_size 1000 \
--max_depth 10 \
--total_floor 10 \
--reduction both \
--sample_config transformer \
--indexes_size 500 \
--stop

python  ./run.py \
--folder_name final_feather \
--num_trees 100 \
--framework finetune \
--index_config per_tree \
--index_config per_tree \
--max_tree_size 1000 \
--max_depth 10 \
--total_floor 10 \
--reduction both \
--sample_config transformer \
--indexes_size 100 \
--stop


python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration bow \
--num_epochs 1 \
--lr 1e-4 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name ignore

python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration lstm \
--num_epochs 1 \
--lr 1 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name ignore

python  ./run.py \
--folder_name final_feather \
--framework finetune \
--skip_setup \
--configuration bow \
--num_epochs 1 \
--lr 1e-4 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name ignore

python  ./run.py \
--folder_name final_feather \
--framework finetune \
--skip_setup \
--configuration lstm \
--num_epochs 1 \
--lr 1 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name ignore


python  ./run.py \
--folder_name final_feather \
--num_trees 100 \
--index_config per_tree \
--indexes_per_tree 5 \
--index_config per_tree \
--max_tree_size 1000 \
--max_depth 10 \
--total_floor 50 \
--reduction both \
--sample_config transformer \
--indexes_size 500 \
--stop

python  ./run.py \
--folder_name final_feather \
--num_trees 100 \
--framework finetune \
--index_config per_tree \
--max_tree_size 1000 \
--max_depth 10 \
--total_floor 50 \
--reduction both \
--sample_config transformer \
--indexes_size 100 \
--stop


python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration bow \
--num_epochs 1 \
--lr 1e-4 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name ignore

python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration lstm \
--num_epochs 1 \
--lr 1 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name ignore

python  ./run.py \
--folder_name final_feather \
--framework finetune \
--skip_setup \
--configuration bow \
--num_epochs 1 \
--lr 1e-4 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name ignore

python  ./run.py \
--folder_name final_feather \
--framework finetune \
--skip_setup \
--configuration lstm \
--num_epochs 1 \
--lr 1 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name ignore




