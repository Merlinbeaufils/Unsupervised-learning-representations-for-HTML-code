#!/bin/bash

export PYTHONPATH=.


python  ./run.py \
--folder_name final_feather \
--num_trees 7500 \
--index_config per_tree \
--indexes_per_tree 5 \
--index_config per_tree \
--max_tree_size 500 \
--max_depth 10 \
--total_floor 100 \
--reduction both \
--sample_config transformer \
--indexes_size 40000 \
--stop

python  ./run.py \
--folder_name final_feather \
--num_trees 1700 \
--framework finetune \
--index_config per_tree \
--max_tree_size 500 \
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
--experiment_name large_run

python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration lstm \
--num_epochs 10 \
--lr 1 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name large_run

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
--experiment_name large_run

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
--experiment_name large_run






python  ./run.py \
--folder_name final_feather \
--num_trees 1000 \
--index_config per_tree \
--indexes_per_tree 5 \
--index_config per_tree \
--max_tree_size 500 \
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
--indexes_per_tree 5 \
--max_tree_size 500 \
--max_depth 10 \
--total_floor 100 \
--reduction both \
--sample_config transformer \
--indexes_size 5000 \
--stop

python  ./run.py \
--folder_name final_feather \
--dont_build_trees \
--index_config per_tree \
--indexes_per_tree 5 \
--total_floor 100 \
--reduction both \
--sample_config transformer \
--indexes_size 5000 \
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
--indexes_size 5000 \
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
--indexes_size 5000 \
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
--indexes_size 5000 \
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
--indexes_size 5000 \
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
--indexes_size 5000 \
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
--indexes_size 5000 \
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
--indexes_size 5000 \
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
--indexes_size 5000 \
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
--indexes_size 5000 \
--data_config key_only \
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
--experiment_name small_run

python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration lstm \
--num_epochs 10 \
--lr 1 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name small_run

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
--experiment_name small_run

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
--experiment_name small_run


python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration bow \
--num_epochs 10 \
--lr 1e-4 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--data_config no_keys \
--experiment_name small_no_keys

python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration lstm \
--num_epochs 10 \
--lr 1 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--data_config no_keys \
--experiment_name small_no_keys

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
--data_config no_keys \
--experiment_name small_no_keys

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
--data_config no_keys \
--experiment_name small_no_keys


python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration bow \
--num_epochs 10 \
--lr 1e-4 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--data_config no_depth \
--experiment_name small_no_depth

python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration lstm \
--num_epochs 10 \
--lr 1 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--data_config no_depth \
--experiment_name small_no_depth

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
--data_config no_depth \
--experiment_name small_no_depth

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
--data_config no_depth \
--experiment_name small_no_depth


python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration bow \
--num_epochs 10 \
--lr 1e-4 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--data_config only_tag \
--experiment_name small_only_tag

python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration lstm \
--num_epochs 10 \
--lr 1 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--data_config only_tag \
--experiment_name small_only_tag

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
--data_config only_tag \
--experiment_name small_only_tag

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
--data_config only_tag \
--experiment_name small_only_tag


python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration bow \
--num_epochs 10 \
--lr 1e-4 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--data_config only_depth \
--experiment_name small_only_depth

python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration lstm \
--num_epochs 10 \
--lr 1 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--data_config only_depth \
--experiment_name small_only_depth

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
--data_config only_depth \
--experiment_name small_only_depth

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
--data_config only_depth \
--experiment_name small_only_depth


python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration bow \
--num_epochs 10 \
--lr 1e-4 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--data_config key_only \
--experiment_name small_key_only

python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration lstm \
--num_epochs 10 \
--lr 1 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--data_config key_only \
--experiment_name small_key_only

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
--data_config key_only \
--experiment_name small_key_only

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
--data_config key_only \
--experiment_name small_key_only



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
--max_tree_size 1000 \
--max_depth 10 \
--total_floor 100 \
--reduction both \
--sample_config transformer \
--indexes_size 5000 \
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
--experiment_name base_1000_trees

python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration lstm \
--num_epochs 10 \
--lr 1 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name base_1000_trees

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
--experiment_name base_1000_trees

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
--experiment_name base_1000_trees



python  ./run.py \
--folder_name final_feather \
--num_trees 1000 \
--index_config per_tree \
--indexes_per_tree 5 \
--index_config per_tree \
--max_tree_size 500 \
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
--max_tree_size 500 \
--max_depth 10 \
--total_floor 100 \
--reduction both \
--sample_config transformer \
--indexes_size 5000 \
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
--experiment_name base_500_trees

python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration lstm \
--num_epochs 10 \
--lr 1 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name base_500_trees

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
--experiment_name base_500_trees

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
--experiment_name base_500_trees


python  ./run.py \
--folder_name final_feather \
--num_trees 1000 \
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
--num_trees 1700 \
--framework finetune \
--index_config per_tree \
--max_tree_size 200 \
--max_depth 10 \
--total_floor 100 \
--reduction both \
--sample_config transformer \
--indexes_size 5000 \
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
--experiment_name base_200_trees

python  ./run.py \
--folder_name final_feather \
--skip_setup \
--configuration lstm \
--num_epochs 10 \
--lr 1 \
--train_proportion 0.8 \
--batch_size 64 \
--embedding_dim 32 \
--experiment_name base_200_trees

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
--experiment_name base_200_trees

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
--experiment_name base_200_trees




