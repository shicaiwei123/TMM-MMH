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

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
parser.add_argument('--parallel', action='store_true', default=False)
parser.add_argument('--workers', type=int, default=2)
parser.add_argument('--mode', default='clip', help='video|clip')
parser.add_argument('--model', default='r50_nl', help='r50|r50_nl')
args = parser.parse_args()


# CV3 train from 916(all):
# SPP
# T=2 51 lambda_kd=1 |  clip  0.912 | video 0.953  啊着,是因为38 训练不够充分的原因吗?
# T=2 101 lambda_kd=0.1| clip 920   |video  0.946
# T=1 38|  clip  0.925 | video 0.944
# CDD
# clip 923| video 0.933


def test(model, test_dataloader):
    net = model
    testloader = test_dataloader
    net.eval()
    add = 0
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

        add += prec_scores[0].cpu()

        for k, prec in zip(topk, prec_scores):
            loss_meters['P%s' % k].add(prec.item(), pred.shape[0])

        stats = ' | '.join(['%s: %.3f' % (k, v.value()[0]) for k, v in loss_meters.items()])
        print('%d/%d.. %s' % (idx, len(testloader), stats))

    print(add / len(testloader))

    print('(test) %s' % stats)


# ----------------------------------------------------------------------------------------------------------------------------------------#
import nw_ucla
from models import resnet

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
    # T=2  0.912 51
    # T=1 0.925 38
    pretrain_dir = "/home/icml//shicaiwei/multi_model_fas/output/models/CDD_CV/_ckpt_E_112_I_2.pth"
    # pretrain_dir = "/home/icml/shicaiwei/multi_model_fas/output/models/multiKD_CV_lambda_kd_max/_ckpt_E_104_I_2.pth"
    # pretrain_dir = "/home/icml//shicaiwei/multi_model_fas/output/models/CDD/_ckpt_E_27_I_2.pth"

    args.split_num = 3

    args.clip_len = 8

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

    net = resnet.i3_res50(10, pretrain_dir=pretrain_dir)

    net.cuda()

    if args.parallel:
        net = nn.DataParallel(net)

    with torch.no_grad():
        test(model=net, test_dataloader=testloader)
