#!/usr/bin/env bash

python surf_baseline_single_main.py '/home/data/shicaiwei/liveness/CASIA-SURF' 'ir' 'resnet18_se_dropout_no_seed' 2 1
python surf_baseline_single_main.py '/home/data/shicaiwei/liveness/CASIA-SURF' 'ir' 'resnet18_se_dropout_no_seed' 2 2
python surf_baseline_single_main.py '/home/data/shicaiwei/liveness/CASIA-SURF' 'ir' 'resnet18_se_dropout_no_seed' 2 3


#python cefa_baseline_single_main.py '/home/data/shicaiwei/cefa/CeFA-Race' 'ir' 'resnet18_se_dropout_no_seed' 1 0
#python cefa_baseline_single_main.py '/home/data/shicaiwei/cefa/CeFA-Race' 'ir' 'resnet18_se_dropout_no_seed' 1 1
#python cefa_baseline_single_main.py '/home/data/shicaiwei/cefa/CeFA-Race' 'ir' 'resnet18_se_dropout_no_seed' 1 2
python cefa_baseline_single_main.py '/home/data/shicaiwei/cefa/CeFA-Race' 'ir' 'resnet18_se_dropout_no_seed' 1 3

python cefa_baseline_single_main.py '/home/data/shicaiwei/cefa/CeFA-Race' 'ir' 'resnet18_se_dropout_no_seed' 1 4
python cefa_baseline_single_main.py '/home/data/shicaiwei/cefa/CeFA-Race' 'ir' 'resnet18_se_dropout_no_seed' 1 5
