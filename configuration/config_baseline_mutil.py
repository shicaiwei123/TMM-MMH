import torchvision.transforms as ts

import torch.optim as optim
import os
import numpy as np
from argparse import ArgumentParser

# GPU

# 训练参数

parser = ArgumentParser()

parser.add_argument('--train_epoch', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lr_decrease', type=str, default='multi_step', help='the methods of learning rate decay  ')
parser.add_argument('--lr_warmup', type=bool, default=True)
parser.add_argument('--total_epoch', type=int, default=10)

parser.add_argument('--weight_decay', type=float, default=1e-3)
parser.add_argument('--momentum', type=float, default=0.90)
parser.add_argument('--class_num', type=int, default=2)
parser.add_argument('--retrain', type=bool, default=False, help='Separate training for the same training process')
parser.add_argument('--log_interval', type=int, default=10, help='How many batches to print the output once')
parser.add_argument('--save_interval', type=int, default=10, help='How many batches to save the model once')
parser.add_argument('--model_root', type=str, default='../output/models')
parser.add_argument('--log_root', type=str, default='../output/logs')
parser.add_argument('--se_reduction', type=int, default=16, help='para for se layer')

parser.add_argument('data_root', type=str,
                    default='/home/bbb/shicaiwei/data/liveness_data/CASIA-SURF')
parser.add_argument('modal', type=str, default='multi')
parser.add_argument('backbone', type=str, default='resnet18_se')
parser.add_argument('gpu', type=int, default=0)

args = parser.parse_args()
args.name = args.backbone + "_" + args.modal
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)