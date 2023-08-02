import sys

sys.path.append('..')
import torch
from itertools import chain
import os

import nw_ucla
from models.model_ensemble_feature import Ensemble
from models.resnet_feature import i3_res50_feature
from loss.kd import *
from lib.model_develop import train_knowledge_distill_action_feature
from configuration.config_ucla_feature_kd import args
import torch.nn as nn


def deeppix_main(args):
    print(args.split_num)
    train_dataset = nw_ucla.NWUCLA_CV(data_root=args.data_root,
                                      split_num=args.split_num,
                                      clip_len=args.clip_len,
                                      mode='all',
                                      train=True,
                                      sample_interal=1)

    test_dataset = nw_ucla.NWUCLA_CV(data_root=args.data_root,
                                     split_num=args.split_num,
                                     clip_len=args.clip_len,
                                     mode='rgb',
                                     train=False,
                                     sample_interal=1)

    nwucla_train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                          shuffle=True,
                                                          num_workers=4,
                                                          pin_memory=False)

    nwucla_test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                         shuffle=True,
                                                         num_workers=4,
                                                         pin_memory=False)

    # seed_torch(2)
    args.log_name = args.name + '.csv'
    args.model_name = args.name

    teacher_model = Ensemble(args.rgb_pretrain_path, args.depth_pretrain_path, args.class_num)
    if args.init_mode == 'kinects':
        student_model = i3_res50_feature(400, "/home/icml/shicaiwei/pytorch-resnet3d/pretrained/i3d_r50_kinetics.pth")
        student_model.fc = nn.Linear(2048, args.class_num)
        student_model.class_num = args.class_num

    elif args.init_mode == 'rgb':
        student_model = i3_res50_feature(args.class_num, args.rgb_pretrain_path)

    elif args.init_mode == 'depth':
        student_model = i3_res50_feature(args.class_num, args.depth_pretrain_path)

    elif args.init_mode == 'load':
        student_model = i3_res50_feature(args.class_num, args.load)

    else:
        student_model = i3_res50_feature(args.class_num, '')



    # 如果有GPU
    if torch.cuda.is_available():
        teacher_model.cuda()  # 将所有的模型参数移动到GPU上
        student_model.cuda()
        print("GPU is using")


    if args.cuda:
        criterionCls = torch.nn.CrossEntropyLoss().cuda()
    else:
        criterionCls = torch.nn.CrossEntropyLoss()

    # initialize optimizer

    if args.optim == 'sgd':
        print('--------------------------------optim with sgd--------------------------------------')


        optimizer = torch.optim.SGD(student_model.parameters(),
                                        lr=args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay,
                                        nesterov=True)
    elif args.optim == 'adam':
        print('--------------------------------optim with adam--------------------------------------')

        optimizer = torch.optim.Adam(student_model.parameters(),
                                         lr=args.lr,
                                         weight_decay=args.weight_decay,
                                         )
    else:
        print('optim error')
        optimizer = None

    # warp nets and criterions for train and test
    nets = {'snet': student_model, 'tnet': teacher_model}
    criterions = criterionCls

    train_knowledge_distill_action_feature(net_dict=nets, criterions=criterions, optimizer=optimizer,
                                   train_loader=nwucla_train_dataloader,
                                   test_loader=nwucla_test_dataloader,
                                   args=args)


if __name__ == '__main__':
    deeppix_main(args)
