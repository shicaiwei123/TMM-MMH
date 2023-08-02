#!/usr/bin/env bash

python cefa_baseline_multi_main.py '/home/data/shicaiwei/cefa/CeFA-Race' 'multi' 0 'cefa_resnet18_no_dropout_no_seed_no_share' 1 0
python cefa_baseline_multi_main.py '/home/data/shicaiwei/cefa/CeFA-Race' 'multi' 0 'cefa_resnet18_no_dropout_no_seed_no_share' 1 1
python cefa_baseline_multi_main.py '/home/data/shicaiwei/cefa/CeFA-Race' 'multi' 0 'cefa_resnet18_no_dropout_no_seed_no_share' 1 2
python cefa_baseline_multi_main.py '/home/data/shicaiwei/cefa/CeFA-Race' 'multi' 0 'cefa_resnet18_no_dropout_no_seed_no_share' 1 3
