import sys

sys.path.append('..')
import torch
from itertools import chain
import os

import ntud60
from models.model_ensemble_learning import Ensemble_SPP_CD
from models.resnet_spp import i3_res50_spp
from loss.kd import *
from lib.model_develop import train_knowledge_distill_action_ensemble
from configuration.config_ucla_ensemble import args
import torch.nn as nn

print(args)


def deeppix_main(args):
    if args.mode == 'video':
        train_dataset = ntud60.NTUD60CS_MultiCrop(data_root=args.data_root,
                                                  clip_len=args.clip_len,
                                                  mode='rgb',
                                                  train=True,
                                                  sample_interal=1)

        test_dataset = ntud60.NTUD60CS_MultiCrop(data_root=args.data_root,
                                                 clip_len=args.clip_len,
                                                 mode='rgb',
                                                 train=False,
                                                 sample_interal=1)

    elif args.mode == 'clip':

        train_dataset = ntud60.NTUD60_CS(data_root=args.data_root,
                                         clip_len=args.clip_len,
                                         mode='rgb',
                                         train=True,
                                         sample_interal=1)

        test_dataset = ntud60.NTUD60_CS(data_root=args.data_root,
                                        clip_len=args.clip_len,
                                        mode='rgb',
                                        train=False,
                                        sample_interal=1)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers)

    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers)

    # seed_torch(2)
    args.log_name = args.name + '.csv'
    args.model_name = args.name

    model = Ensemble_SPP_CD(args=args, spp_pretrained_dir=args.spp_pretrained_dir,
                            cp_pretrained_dir=args.cd_pretrained_dir)

    # 如果有GPU
    if torch.cuda.is_available():
        model.cuda()  # 将所有的模型参数移动到GPU上
        print("GPU is using")

    # initialize optimizer

    if args.optim == 'sgd':

        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    nesterov=True)
    elif args.optim == 'adam':
        print('--------------------------------optim with adam--------------------------------------')

        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay,
                                     )
    else:
        print('optim error')
        optimizer = None

    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        criterion = criterion.cuda()

    train_knowledge_distill_action_ensemble(model=model, cost=criterion, optimizer=optimizer, testloader=testloader,
                                            train_loader=trainloader,
                                            args=args)


if __name__ == '__main__':
    deeppix_main(args)
