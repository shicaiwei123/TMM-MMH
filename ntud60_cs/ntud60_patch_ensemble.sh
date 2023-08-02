#!/usr/bin/env bash
python ntud60_patch_ensemble_main.py '/home/CVPR/shicaiwei/multi_model_fas/output/models/CDD_ntud60/_ckpt_E_68_I_1.pth' '/home/CVPR/shicaiwei/multi_model_fas/output/models/multiKD_lambda_kd/_ckpt_E_68_I_1.pth'  1 0 1 'clip'

#python ucla_patch_ensemble_main.py '/home/icml//shicaiwei/multi_model_fas/output/models/multiKD/_ckpt_E_290_I_0.pth' '/home/icml/shicaiwei/multi_model_fas/output/models/ckpt_E_224_I_1.pth'  0 1 1 'video'
