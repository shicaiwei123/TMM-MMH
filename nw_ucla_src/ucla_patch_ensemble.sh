#!/usr/bin/env bash
#python ucla_patch_ensemble_main.py '/home/icml//shicaiwei/multi_model_fas/output/models/multiKD/_ckpt_E_290_I_0.pth' '/home/icml/shicaiwei/multi_model_fas/output/models/ckpt_E_224_I_1.pth'  0 0 1 'clip'

python ucla_patch_ensemble_main.py '/home/icml//shicaiwei/multi_model_fas/output/models/multiKD/_ckpt_E_290_I_0.pth' '/home/icml/shicaiwei/multi_model_fas/output/models/ckpt_E_224_I_1.pth'  0 1 1 'video'
