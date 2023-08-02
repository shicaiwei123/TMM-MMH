import sys

sys.path.append('..')
import torch
from itertools import chain
import os

from data import ucf101
from models.model_ensemble import Ensemble_I3D
from models import i3dpt_spp
from loss.kd import *
from lib.model_develop import train_knowledge_distill_action
from configuration.config_ucf101_spp import args
import torch.nn as nn


def deeppix_main(args):
    nwucla_trainset = ucf101.UCF101(data_root="/home/ssd/video_action/ucf101/rawframes_resize",
                                    split_path="../data/split_divide/ucf101/trainlist01.txt",
                                    clip_len=args.clip_len,
                                    mode='all',
                                    sample_interal=1)

    nwucla_testset = ucf101.UCF101(data_root="/home/ssd/video_action/ucf101/rawframes_resize",
                                   clip_len=args.clip_len,
                                   split_path="../data/split_divide/ucf101/testlist_1.txt",
                                   mode='all',
                                   sample_interal=1)

    trainloader = torch.utils.data.DataLoader(nwucla_trainset, batch_size=args.batch_size, shuffle=True,
                                              num_workers=args.workers,
                                              pin_memory=False)

    testloader = torch.utils.data.DataLoader(nwucla_testset, batch_size=args.batch_size*2, shuffle=True,
                                             num_workers=args.workers,
                                             pin_memory=False)

    # seed_torch(2)
    args.log_name = args.name + '.csv'
    args.model_name = args.name

    teacher_model = Ensemble_I3D(args.rgb_pretrain_path, args.flow_pretrain_path, args.class_num)
    for p in teacher_model.parameters():
        p.requires_grad = False
    teacher_model.eval()
    if args.init_mode == 'kinects':
        student_model = i3dpt_spp.I3D_SPP(args.class_num, modality='rgb')
        student_model.conv3d_0c_1x1 = i3dpt_spp.Unit3Dpy(
            in_channels=1024,
            out_channels=400,
            kernel_size=(1, 1, 1),
            activation=None,
            use_bias=True,
            use_bn=False)
        student_model.load_state_dict(torch.load("../pretrained/model_rgb.pth"))
        student_model.conv3d_0c_1x1 = i3dpt_spp.Unit3Dpy(
            in_channels=1024,
            out_channels=args.class_num,
            kernel_size=(1, 1, 1),
            activation=None,
            use_bias=True,
            use_bn=False)

    elif args.init_mode == 'rgb':
        student_model = i3dpt_spp.I3D_SPP(args.class_num, modality='rgb')
        student_model.load_state_dict(torch.load(args.rgb_pretrain_path)['net'])

        student_model.conv3d_0c_1x1 = i3dpt_spp.Unit3Dpy(
            in_channels=1024,
            out_channels=args.class_num,
            kernel_size=(1, 1, 1),
            activation=None,
            use_bias=True,
            use_bn=False)

    elif args.init_mode == 'flow':
        student_model = i3dpt_spp.I3D_SPP(args.class_num, modality='rgb')
        pretrain_para = torch.load(args.flow_pretrain_path)['net']
        error_keys = []
        for k, v in pretrain_para.items():
            if student_model.state_dict()[k].shape != v.shape:
                error_keys.append(k)
                print(k)
        for k in error_keys:
            del pretrain_para[k]
        student_model.load_state_dict(pretrain_para, strict=False)

        student_model.conv3d_0c_1x1 = i3dpt_spp.Unit3Dpy(
            in_channels=1024,
            out_channels=args.class_num,
            kernel_size=(1, 1, 1),
            activation=None,
            use_bias=True,
            use_bn=False)
    else:
        print("random")

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
        if args.kd_mode in ['vid', 'ofd', 'afd', 'multi_st']:
            optimizer = torch.optim.SGD(chain(student_model.parameters(), criterionKD.parameters()),
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
        if args.kd_mode in ['vid', 'ofd', 'afd', 'multi_st']:
            optimizer = torch.optim.Adam(chain(student_model.parameters(), criterionKD.parameters()),
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
                                   train_loader=trainloader,
                                   test_loader=testloader,
                                   args=args)


if __name__ == '__main__':
    deeppix_main(args)
