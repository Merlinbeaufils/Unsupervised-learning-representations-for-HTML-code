#!/bin/bash

export PYTHONPATH=C:/Users/merli/PycharmProjects/Thesis
python  C:/Users/merli/PycharmProjects/Thesis/run.py \
--folder_name feather \
--num_trees 100 \
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

