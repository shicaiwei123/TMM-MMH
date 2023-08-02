import torchvision.transforms as ts

import torch.optim as optim
import os
import numpy as np
from argparse import ArgumentParser

# 训练参数

parser = ArgumentParser()

parser.add_argument('--train_epoch', type=int, default=40)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decrease', type=str, default='multi_step', help='the methods of learning rate decay  ')
parser.add_argument('--lr_warmup', type=bool, default=False)
parser.add_argument('--total_epoch', type=int, default=10)

parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--momentum', type=float, default=0.90)
parser.add_argument('--class_num', type=int, default=2)
parser.add_argument('--retrain', type=bool, default=False, help='Separate training for the same training process')
parser.add_argument('--log_interval', type=int, default=10, help='How many batches to print the output once')
parser.add_argument('--save_interval', type=int, default=5, help='How many epochs to save the model once')
parser.add_argument('--model_root', type=str, default='../output/models')
parser.add_argument('--log_root', type=str, default='../output/logs')
parser.add_argument('--se_reduction', type=int, default=16, help='para for se layer')
parser.add_argument('--optim', type=str, default='sgd')
parser.add_argument('--method', type=str, default='admd_kd')

parser.add_argument('--teacher_modal', type=str, default='depth', help="the origin modal to train the teacher")
parser.add_argument('--teacher_data', type=str, default='depth',
                    help='data for teacher or the data knowledge you want to transfer:multi,depth,ir')
parser.add_argument('--student_modal', type=str, default='rgb', help='the origin modal to train the student ')
parser.add_argument('--student_data', type=str, default='rgb',
                    help='the data to trained the student modal: multi_rgb,single_rgb')
parser.add_argument('--init_mode', type=str, default='depth',
                    help='the way to init the student net: random, rgb, depth, ir,sp_feature')

parser.add_argument('--student_name', type=str, default='resnet18_se', help='the backbone for student: resnet18_se')

parser.add_argument('--lambda_kd', type=float, default=1.0, help='trade-off parameter for kd loss')
parser.add_argument('--cuda', type=bool, default=True)

parser.add_argument('--generate_iternum', type=int, default=1, help='the iter times in each epoch')
parser.add_argument('--discriminator_iternum', type=int, default=1, help='the iter times in each epoch')

parser.add_argument('--feature_dim', type=int, default=512)
parser.add_argument('data_root', type=str,
                    default='/home/shicaiwei/data/liveness_data/CASIA-SUFR')
parser.add_argument('gpu', type=int, default=0)
parser.add_argument('version', type=int, default=2)

args = parser.parse_args()
args.name = args.method + "_" + args.teacher_data + "_" + args.student_data + "_lr_" + str(args.lr)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

# params for optimizing models
d_learning_rate = 1e-4
c_learning_rate = 1e-4
beta1 = 0.5
beta2 = 0.9

args.d_learning_rate = d_learning_rate
args.c_learning_rate = c_learning_rate
args.beta1 = beta1
args.beta2 = beta2
