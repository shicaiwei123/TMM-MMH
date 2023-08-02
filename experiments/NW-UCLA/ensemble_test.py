import torch
import torch.nn as nn
import numpy as np
import argparse
import collections
import torchnet as tnt
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
sys.path.append('..')
from action_utils import util

from configuration.config_ucla_ensemble import args


# CV3-linear
# 33  | clip 0.918  | video 944
# 90  | clip 0.918  | video 940 (3crop) 942 (val)

#CV-simple add
#    | clip 929    | video 0.948

def test(model, test_dataloader):
    print(1)
    net = model
    testloader = test_dataloader
    net.eval()

    topk = [1, 5]
    loss_meters = collections.defaultdict(lambda: tnt.meter.AverageValueMeter())
    for idx, batch in enumerate(testloader):
        print(idx)
        print(1)

        batch = util.batch_cuda(batch)
        pred, loss_dict = net(batch)

        loss_dict = {k: v.mean() for k, v in loss_dict.items() if v.numel() > 0}
        loss = sum(loss_dict.values())

        for k, v in loss_dict.items():
            loss_meters[k].add(v.item())

        print(idx)
        if idx>25:
            print(pred,batch['label'])

        prec_scores = util.accuracy(pred, batch['label'], topk=topk)
        for k, prec in zip(topk, prec_scores):
            loss_meters['P%s' % k].add(prec.item(), pred.shape[0])

        stats = ' | '.join(['%s: %.3f' % (k, v.value()[0]) for k, v in loss_meters.items()])
        print('%d/%d.. %s' % (idx, len(testloader), stats))

    print('(test) %s' % stats)


# ----------------------------------------------------------------------------------------------------------------------------------------#
import nw_ucla
from models.model_ensemble_learning import Ensemble_SPP_CD

if __name__ == '__main__':

    pretrain_dir =    "/home/icml//shicaiwei/multi_model_fas/output/models/spp_ensemble_cd_cv/_ckpt_E_33_I_2.pth"

    args.split_num = 3
    args.clip_len = 8
    args.spp_pretrained_dir = "/home/icml//shicaiwei/multi_model_fas/output/models/multiKD_CV_lambda_kd/_ckpt_E_110_I_2.pth"
    args.cd_pretrained_dir = "/home/icml//shicaiwei/multi_model_fas/output/models/CDD_CV/_ckpt_E_112_I_2.pth"
    args.gpu = 0
    args.version = 1
    lr_warmup = 1
    args.mode = 'video'

    print(args.split_num)
    if args.mode == 'video':
        nwucla_testset = nw_ucla.NWUCLACV_MultiCrop(data_root="/home/data/shicaiwei/N-UCLA/multiview_action",
                                                    split_num=args.split_num,
                                                    clip_len=args.clip_len,
                                                    mode='rgb',
                                                    train=False,
                                                    sample_interal=1)
    elif args.mode == 'clip':

        nwucla_testset = nw_ucla.NWUCLA_CV(data_root="/home/data/shicaiwei/N-UCLA/multiview_action",
                                           split_num=args.split_num,
                                           clip_len=args.clip_len,
                                           mode='rgb',
                                           train=False,
                                           sample_interal=1)

    testloader = torch.utils.data.DataLoader(nwucla_testset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers)

    net = Ensemble_SPP_CD(args=args, spp_pretrained_dir=args.spp_pretrained_dir,
                          cp_pretrained_dir=args.cd_pretrained_dir)

    net.load_state_dict(torch.load(pretrain_dir)['net'])

    net.cuda()

    print(1)
    with torch.no_grad():
        print(1)
        test(model=net, test_dataloader=testloader)
