import sys

sys.path.append('..')
from models.surf_pyramid import SURF_Pyramid
from src.surf_pyramid_dataloader import surf_pyramid_dataloader, surf_pyramid_transforms_train
from configuration.config_pyramid_multi import args
import torch
import torch.nn as nn
from lib.model_develop import train_multi_advsor
import torch.optim as optim

import cv2
import numpy as np
import datetime
import random

'''
TO DO:
debug resnet 预训练参数加载
debug resnet 底层层数设计,特别是make layer 层,弄明白模型参数重载的原理
'''

args.enhancement = surf_pyramid_transforms_train


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def pyramid_main(args):
    train_loader = surf_pyramid_dataloader(train=True, args=args)
    test_loader = surf_pyramid_dataloader(train=False, args=args)

    # seed_torch(2)
    args.log_name = args.name + '.csv'
    args.model_name = args.name

    model = SURF_Pyramid(args, pretrained=args.pretrained)
    # 如果有GPU
    if torch.cuda.is_available():
        model.cuda()  # 将所有的模型参数移动到GPU上
        print("GPU is using")

    criterion = nn.BCELoss()

    if args.optim=='sgd':

        optimizer = optim.SGD(filter(lambda param: param.requires_grad, model.parameters()), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    else:
        optimizer = optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr=args.lr,
                               weight_decay=args.weight_decay)

    args.retrain = False
    train_multi_advsor(model=model, cost=criterion, optimizer=optimizer, train_loader=train_loader,
                       test_loader=test_loader,
                       args=args)


if __name__ == '__main__':
    pyramid_main(args=args)
