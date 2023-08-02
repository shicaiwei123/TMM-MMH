#!/usr/bin/env bash
python ucla_patch_ensemble_main.py "/home/icml//shicaiwei/multi_model_fas/output/models/multiKD_CV_lambda_kd/_ckpt_E_110_I_2.pth" "/home/icml//shicaiwei/multi_model_fas/output/models/CDD_CV/_ckpt_E_94_I_2.pth"  0 0 1 'clip'
#
#python ucla_patch_ensemble_main.py "/home/icml//shicaiwei/multi_model_fas/output/models/multiKD_CV_lambda_kd/_ckpt_E_110_I_2.pth" "/home/icml//shicaiwei/multi_model_fas/output/models/CDD_CV/_ckpt_E_94_I_2.pth"  0 1 1 'video'
