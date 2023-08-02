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

#simple add
# clip 80.1 | video 85.6 可以了

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
import ntud60
from models.model_ensemble_learning import Ensemble_SPP_CD

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
    pretrain_dir = "/home/icml//shicaiwei/multi_model_fas/output/models/spp_ensemble_cd/_ckpt_E_109_I_1.pth"

    args.clip_len = 16
    args.spp_pretrained_dir = "/home/CVPR/shicaiwei/multi_model_fas/output/models/multiKD_lambda_kd_new/_ckpt_E_142_I_2.pth"
    args.cd_pretrained_dir = "/home/CVPR/shicaiwei/multi_model_fas/output/models/CDD_ntud60/_ckpt_E_68_I_1.pth"
    args.gpu = 1
    args.version = 1
    lr_warmup = 1
    args.mode = 'video'

    if args.mode == 'video':
        test_dataset = ntud60.NTUD60CS_MultiCrop(data_root=args.data_root,
                                                 clip_len=args.clip_len,
                                                 mode='rgb',
                                                 train=False,
                                                 sample_interal=1)
    elif args.mode == 'clip':

        test_dataset = ntud60.NTUD60_CS(data_root=args.data_root,
                                        clip_len=args.clip_len,
                                        mode='rgb',
                                        train=False,
                                        sample_interal=1)

    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers)

    net = Ensemble_SPP_CD(args=args, spp_pretrained_dir=args.spp_pretrained_dir,
                          cp_pretrained_dir=args.cd_pretrained_dir)

    # net.load_state_dict(torch.load(pretrain_dir)['net'])

    net.cuda()

    with torch.no_grad():
        test(model=net, test_dataloader=testloader)
