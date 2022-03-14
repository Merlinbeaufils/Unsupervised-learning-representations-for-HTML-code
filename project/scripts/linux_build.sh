#!/bin/bash
#SBATCH --job-name=exp
#SBATCH --gres=gpu:4
#SBATCH --qos=qos_gpu-t4
#SBATCH --cpus-per-task=4
#SBATCH --output=./exp.out
#SBATCH --error=./exp.err
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1


srun --chdir ~/Thesis  python3 run.py \
--folder_name feather \
--num_trees 10000 \
--index_config per_tree \
--indexes_per_tree 50 \
--index_config per_tree \
--max_tree_size 100 \
--max_depth 4 \
--key_only \
--reduction both \
--sample_config transformer \
--indexes_size 5000 \
--stop

