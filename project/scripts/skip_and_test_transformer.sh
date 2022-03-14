#!/bin/bash

export PYTHONPATH=C:/Users/merli/PycharmProjects/Thesis
python  C:/Users/merli/PycharmProjects/Thesis/run.py \
--folder_name feather \
--skip_setup \
--configuration transformer \
--num_epochs 10 \
--run_name attempt_small \
--train_proportion 0.6 \
--batch_size 5 \
--lr 5e-4 \
--weight_decay 1e-4 \
--dataloader transformer

