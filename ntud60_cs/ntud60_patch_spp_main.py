import sys

sys.path.append('..')
import torch
from itertools import chain
import os

import ntud60
from models.model_ensemble import Ensemble
from models.resnet_spp import i3_res50_spp
from loss.kd import *
from lib.model_develop import train_knowledge_distill_action
from configuration.config_ucla_spp import args
import torch.nn as nn


def deeppix_main(args):
    train_dataset = ntud60.NTUD60_CS(data_root=args.data_root,
                                      clip_len=args.clip_len,
                                      mode='all',
                                      train=True,
                                      sample_interal=1)

    test_dataset = ntud60.NTUD60_CS(data_root=args.data_root,
                                     clip_len=args.clip_len,
                                     mode='rgb',
                                     train=False,
                                     sample_interal=1)

    nwucla_train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                          shuffle=True,
                                                          num_workers=4,
                                                          pin_memory=False)

    nwucla_test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                         shuffle=False,
                                                         num_workers=4,
                                                         pin_memory=False)

    # seed_torch(2)
    args.log_name = args.name + '.csv'
    args.model_name = args.name

    teacher_model = Ensemble(args.rgb_pretrain_path, args.depth_pretrain_path, args.class_num)
    if args.init_mode == 'kinects':
        student_model = i3_res50_spp(400, "/home/CVPR/shicaiwei/pytorch-resnet3d/pretrained/i3d_r50_kinetics.pth")
        student_model.fc = nn.Linear(2048, args.class_num)
        student_model.class_num = args.class_num

    elif args.init_mode == 'rgb':
        student_model = i3_res50_spp(args.class_num, args.rgb_pretrain_path)

    elif args.init_mode == 'depth':
        student_model = i3_res50_spp(args.class_num, args.depth_pretrain_path)

    else:
        student_model = i3_res50_spp(args.class_num, '')

    if args.fc_mode == 'random':
        print(
            '--------------------------------init student_fc with random--------------------------------------')
        student_model.fc = nn.Linear(2048, args.class_num)

    if args.load != '':
        student_model = i3_res50_spp(args.class_num, args.load)

    # 如果有GPU
    if torch.cuda.is_available():
        teacher_model.cuda()  # 将所有的模型参数移动到GPU上
        student_model.cuda()
        print("GPU is using")

        # define loss functions
    if args.kd_mode == 'logits':
        criterionKD = Logits()
    elif args.kd_mode == 'st':
        criterionKD = SoftTarget(args.T)
    elif args.kd_mode == 'at':
        criterionKD = AT(args.p)
    elif args.kd_mode == 'fitnet':
        criterionKD = Hint()
    elif args.kd_mode == "multi_st":
        criterionKD = MultiSoftTarget(args.T)

    else:
        raise Exception('Invalid kd mode...')
    if args.cuda:
        criterionCls = torch.nn.CrossEntropyLoss().cuda()
        criterionKD = criterionKD.cuda()
    else:
        criterionCls = torch.nn.CrossEntropyLoss()

    # initialize optimizer

    if args.optim == 'sgd':
        print('--------------------------------optim with sgd--------------------------------------')
        if args.kd_mode in ['vid', 'ofd', 'afd']:
            optimizer = torch.optim.SGD(chain(student_model.parameters(),
                                              *[c.parameters() for c in criterionKD[1:]]),
                                        lr=args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay,
                                        nesterov=True)
        else:
            optimizer = torch.optim.SGD(filter(lambda param: param.requires_grad, student_model.parameters()),
                                        lr=args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay,
                                        nesterov=True)
    elif args.optim == 'adam':
        print('--------------------------------optim with adam--------------------------------------')
        if args.kd_mode in ['vid', 'ofd', 'afd']:
            optimizer = torch.optim.Adam(chain(student_model.parameters(),
                                               *[c.parameters() for c in criterionKD[1:]]),
                                         lr=args.lr,
                                         weight_decay=args.weight_decay,
                                         )
        else:
            optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, student_model.parameters()),
                                         lr=args.lr,
                                         weight_decay=args.weight_decay,
                                         )
    else:
        print('optim error')
        optimizer = None

    # warp nets and criterions for train and test
    nets = {'snet': student_model, 'tnet': teacher_model}
    criterions = {'criterionCls': criterionCls, 'criterionKD': criterionKD}

    train_knowledge_distill_action(net_dict=nets, cost_dict=criterions, optimizer=optimizer,
                                   train_loader=nwucla_train_dataloader,
                                   test_loader=nwucla_test_dataloader,
                                   args=args)


if __name__ == '__main__':
    deeppix_main(args)
