#!/usr/bin/env bash
python ucla_patch_feature_main.py '/home/icml//shicaiwei/pytorch-resnet3d/src/cv/tmp/rgb/3/8/ckpt_clip3_E_254_I_2.pth' '/home/icml//shicaiwei/pytorch-resnet3d/src/cv/tmp/depth/3/8/ckpt_clip3_E_227_I_2.pth'  0 0 1 100
python ucla_patch_feature_main.py '/home/icml//shicaiwei/pytorch-resnet3d/src/cv/tmp/rgb/3/8/ckpt_clip3_E_254_I_2.pth' '/home/icml//shicaiwei/pytorch-resnet3d/src/cv/tmp/depth/3/8/ckpt_clip3_E_227_I_2.pth'  0 1 1 100

