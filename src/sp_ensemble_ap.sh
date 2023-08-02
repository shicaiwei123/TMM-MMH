#!/usr/bin/env bash

#python sp_ensemble_ap_main.py '/home/bbb/shicaiwei/data/liveness_data/CASIA-SURF' 0 0 0   #best
#python sp_ensemble_ap_main.py '/home/bbb/shicaiwei/data/liveness_data/CASIA-SURF' 0 1 1
#python sp_ensemble_ap_main.py '/home/bbb/shicaiwei/data/liveness_data/CASIA-SURF' 0 2 0   # worst
#python sp_ensemble_ap_main.py '/home/bbb/shicaiwei/data/liveness_data/CASIA-SURF' 0 3 1

python sp_ensemble_ap_main.py '/home/bbb/shicaiwei/data/liveness_data/CASIA-SURF' 'patch_kd_multiKD_multi_multi_rgb_lr_0.001_version_4_weight_patch_0_select_0_acer_best_.pth' 'patch_feature_kd_mmd_avg_sp_multi_multi_rgb_lr_0.001_version_7_sift_0_acer_best_.pth' 0 8 0   # worst 0.5 0.5
python sp_ensemble_ap_main.py '/home/bbb/shicaiwei/data/liveness_data/CASIA-SURF' 'patch_kd_multiKD_multi_multi_rgb_lr_0.001_version_4_weight_patch_0_select_0_acer_best_.pth' 'patch_feature_kd_mmd_avg_sp_multi_multi_rgb_lr_0.001_version_7_sift_0_acer_best_.pth' 0 9 1

#python sp_ensemble_ap_main.py '/home/bbb/shicaiwei/data/liveness_data/CASIA-SURF' 1 4 1  # best linear
#python sp_ensemble_ap_main.py '/home/bbb/shicaiwei/data/liveness_data/CASIA-SURF' 1 5 1
#python sp_ensemble_ap_main.py '/home/bbb/shicaiwei/data/liveness_data/CASIA-SURF' 1 6 1
#python sp_ensemble_ap_main.py '/home/bbb/shicaiwei/data/liveness_data/CASIA-SURF' 1 7 1

#python sp_ensemble_ap_main.py '/home/bbb/shicaiwei/data/liveness_data/CASIA-SURF' 0 4 1
#python sp_ensemble_ap_main.py '/home/bbb/shicaiwei/data/liveness_data/CASIA-SURF' 0 5 1
#python sp_ensemble_ap_main.py '/home/bbb/shicaiwei/data/liveness_data/CASIA-SURF' 0 6 1
#python sp_ensemble_ap_main.py '/home/bbb/shicaiwei/data/liveness_data/CASIA-SURF' 0 7 1