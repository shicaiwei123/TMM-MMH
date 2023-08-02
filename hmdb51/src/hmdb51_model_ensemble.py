import sys

sys.path.append('..')
import torch
from itertools import chain
import os

from data import hmdb51
from models import model_ensemble
from loss.kd import *
from lib.model_develop import train_knowledge_distill_action_ensemble
from configuration.config_hmdb51_ensemble import args
import torch.nn as nn


def deeppix_main(args):
    nwucla_trainset = hmdb51.HMDB51(data_root="/home/ssd/video_action/hmdb51/rawframes_resize",
                                    split_path="../data/split_divide/hmdb/train_list1.txt",
                                    clip_len=args.clip_len,
                                    mode='all',
                                    sample_interal=1)

    nwucla_testset = hmdb51.HMDB51(data_root="/home/ssd/video_action/hmdb51/rawframes_resize",
                                   clip_len=args.clip_len,
                                   split_path="../data/split_divide/hmdb/test_list1.txt",
                                   mode='all',
                                   sample_interal=1)

    trainloader = torch.utils.data.DataLoader(nwucla_trainset, batch_size=args.batch_size, shuffle=True,
                                              num_workers=args.workers,
                                              pin_memory=False)

    testloader = torch.utils.data.DataLoader(nwucla_testset, batch_size=args.batch_size, shuffle=True,
                                             num_workers=args.workers,
                                             pin_memory=False)

    # seed_torch(2)
    args.log_name = args.name + '.csv'
    args.model_name = args.name

    # model and init
    ensenble_model = model_ensemble.Ensemble_I3D_RGB(args.rgb_pretrain_path, args.transfer_pretain_dir, args.class_num)
    ensenble_model=ensenble_model.cuda()


    if args.optim == 'sgd':

        optimizer = torch.optim.SGD(filter(lambda param: param.requires_grad, ensenble_model.parameters()),
                                        lr=args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay,
                                        nesterov=True)
    elif args.optim == 'adam':

        optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, ensenble_model.parameters()),
                                         lr=args.lr,
                                         weight_decay=args.weight_decay,
                                         )
    else:
        print('optim error')
        optimizer = None

    train_knowledge_distill_action_ensemble(ensenble_model, optimizer=optimizer,
                                          train_loader=trainloader,
                                          test_loader=testloader,
                                          args=args)


if __name__ == '__main__':
    deeppix_main(args)
