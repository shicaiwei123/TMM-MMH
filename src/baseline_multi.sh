#!/usr/bin/env bash

#python surf_baseline_multi_main.py '/home/data/shicaiwei/liveness/CASIA-SURF' 'multi' 0 'resnet18_no_dropout_no_seed_no_share_new' 1 0
#python surf_baseline_multi_main.py '/home/data/shicaiwei/liveness/CASIA-SURF' 'multi' 0 'resnet18_no_dropout_no_seed_no_share_new' 1 1
#python surf_baseline_multi_main.py '/home/data/shicaiwei/liveness/CASIA-SURF' 'multi' 0 'resnet18_no_dropout_no_seed_no_share_new' 1 2
#python surf_baseline_multi_main.py '/home/data/shicaiwei/liveness/CASIA-SURF' 'multi' 0 'resnet18_no_dropout_no_seed_no_share_new' 1 3
#


python cefa_baseline_multi_main.py '/home/data/shicaiwei/cefa/CeFA-Race' 'multi' 0 'cefa_resnet18_no_dropout_no_seed_no_share' 1 0
python cefa_baseline_multi_main.py '/home/data/shicaiwei/cefa/CeFA-Race' 'multi' 0 'cefa_resnet18_no_dropout_no_seed_no_share' 1 1
python cefa_baseline_multi_main.py '/home/data/shicaiwei/cefa/CeFA-Race' 'multi' 0 'resnet18_no_dropout_no_seed_no_share' 1 2
python cefa_baseline_multi_main.py '/home/data/shicaiwei/cefa/CeFA-Race' 'multi' 0 'resnet18_no_dropout_no_seed_no_share' 1 3



