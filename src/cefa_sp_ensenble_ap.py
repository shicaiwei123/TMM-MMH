import sys
sys.path.append('..')
import torch
from itertools import chain
import os

from src.multimodal_to_rgb_kd_dataloader import multimodal_to_rgb_kd_dataloader
from src.cefa_baseline_multi_dataloader import cefa_baseline_multi_dataloader
from models.sp_ap_ensemble import SAP
from loss.kd import *
from lib.model_develop import train_ensemble
from configuration.config_sp_ap_ensemble import args
from lib.processing_utils import seed_torch


def deeppix_main(args):

    # pair is not need
    train_loader = cefa_baseline_multi_dataloader(train=True, args=args)
    test_loader = cefa_baseline_multi_dataloader(train=False, args=args)


    # seed_torch(2)
    args.log_name = args.name + '.csv'
    args.model_name = args.name
    sp_pretrained_path = os.path.join(args.model_root,
                                      args.sp_model)
    ap_pretrained_path = os.path.join(args.model_root,
                                      args.ap_model)
    model = SAP(args=args, sp_pretrained_dir=sp_pretrained_path, ap_pretrained_dir=ap_pretrained_path)

    # 如果有GPU
    if torch.cuda.is_available():
        model = model.cuda()  # 将所有的模型参数移动到GPU上
        print("GPU is using")

    if args.cuda:
        criterionCls = torch.nn.CrossEntropyLoss().cuda()
    else:
        criterionCls = torch.nn.CrossEntropyLoss()

    # initialize optimizer

    if args.optim == 'sgd':
        print('--------------------------------optim with sgd--------------------------------------')

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

    train_ensemble(model=model, cost=criterionCls, optimizer=optimizer, train_loader=train_loader,
                   test_loader=test_loader,
                   args=args)


if __name__ == '__main__':
    deeppix_main(args)
