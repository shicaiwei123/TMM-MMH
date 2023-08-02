#!/usr/bin/env bash
python ucla_patch_feature_main.py '/home/icml/shicaiwei/pytorch-resnet3d/src/cs/tmp/rgb/10/16/ckpt_clip10_E_226_I_1.pth' '/home/icml/shicaiwei/pytorch-resnet3d/src/cs/tmp/depth/10/16/ckpt_clip10_E_308_I_1.pth'  1 0 1 100
python ucla_patch_feature_main.py '/home/icml/shicaiwei/pytorch-resnet3d/src/cs/tmp/rgb/10/16/ckpt_clip10_E_226_I_1.pth' '/home/icml/shicaiwei/pytorch-resnet3d/src/cs/tmp/depth/10/16/ckpt_clip10_E_308_I_1.pth'  1 1 1 100

