import torch
import torch.nn as nn
import numpy as np
import argparse
import collections
import torchnet as tnt
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
sys.path.append('..')
from action_utils import util

from configuration.config_ucla_ensemble import args


# suject 10


def test(model, test_dataloader):
    net = model
    testloader = test_dataloader
    net.eval()

    topk = [1, 5]
    loss_meters = collections.defaultdict(lambda: tnt.meter.AverageValueMeter())
    for idx, batch in enumerate(testloader):

        batch = util.batch_cuda(batch)
        pred, loss_dict = net(batch)

        loss_dict = {k: v.mean() for k, v in loss_dict.items() if v.numel() > 0}
        loss = sum(loss_dict.values())

        for k, v in loss_dict.items():
            loss_meters[k].add(v.item())

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

    os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
    pretrain_dir = "/home/icml//shicaiwei/multi_model_fas/output/models/spp_ensemble_cd/_ckpt_E_109_I_1.pth"

    args.split_subject = 10
    args.clip_len = 16
    args.spp_pretrained_dir = "/home/icml//shicaiwei/multi_model_fas/output/models/multiKD/_ckpt_E_290_I_0.pth"
    args.cd_pretrained_dir = "/home/icml/shicaiwei/multi_model_fas/output/models/ckpt_E_224_I_1.pth"
    args.gpu = 1
    args.version = 1
    lr_warmup = 1
    args.mode = 'video'

    print(args.split_subject)
    if args.mode == 'video':
        nwucla_testset = nw_ucla.NWUCLACS_MultiCrop(data_root="/home/data/shicaiwei/N-UCLA/multiview_action",
                                                    split_subject=args.split_subject,
                                                    clip_len=args.clip_len,
                                                    mode='rgb',
                                                    train=False,
                                                    sample_interal=1)
    elif args.mode == 'clip':

        nwucla_testset = nw_ucla.NWUCLA_CS(data_root="/home/data/shicaiwei/N-UCLA/multiview_action",
                                           split_subject=args.split_subject,
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


    with torch.no_grad():
        test(model=net, test_dataloader=testloader)
