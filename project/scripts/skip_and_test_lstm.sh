#!/bin/bash

export PYTHONPATH=C:/Users/merli/PycharmProjects/Thesis
python  C:/Users/merli/PycharmProjects/Thesis/run.py \
--folder_name feather \
--skip_setup \
--configuration lstm \
--num_epochs 10 \
--lr 1 \
--run_name attempt_small \
--train_proportion 0.8 \
--batch_size 10
