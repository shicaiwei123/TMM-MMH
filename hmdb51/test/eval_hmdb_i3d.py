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
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--mode', default='rgb', help='video|clip')
parser.add_argument('--clip', default='clip', help='video|clip')
parser.add_argument('--model', default='r50_nl', help='r50|r50_nl')
parser.add_argument('--gpu', default=2, type=int)
args = parser.parse_args()
if args.clip=='video':
    args.batch_size=4


# suject
# Depth test in icml (154, train 90.2) | clip 78.3 | video 85.5|   --- (248,train,94.3)| clip 79.1 |video 86.2
# RGB test in cvpr    (197, train 93.4)| clip 75.4| video  0.819
# ALL in CVPR

# clip 8
# rgb |clip 76.2  |video


def test(model, test_dataloader):
    net = model
    testloader = test_dataloader
    net.eval()

    topk = [1, 5]
    loss_meters = collections.defaultdict(lambda: tnt.meter.AverageValueMeter())
    for idx, batch in enumerate(testloader):

        batch = util.batch_cuda(batch)
        result = net(batch)
        if isinstance(result, tuple):
            if len(result) == 3:
                pred = result[0]
                loss_dict = result[2]
            elif len(result) == 2:
                pred = result[0]
                loss_dict = result[1]

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
    for k, v in loss_meters.items():
        if k == 'P1':
            socre = v.value()[0]


# ----------------------------------------------------------------------------------------------------------------------------------------#
from data import hmdb51
from models import i3dpt_spp

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
    # pretrain_dir = "../results/hmdb51_0.01/0/rgb/16/ckpt_E_94_I_1.pth"
    pretrain_dir = "../results/dask/hmdb51_0.01/0/all/kinects/lambda_kd_2.0/weight_patch_0/_ckpt_E_115_I_2.pth"

    clip_len = 16
    args.clip_len = clip_len

    img_mode = 'all'
    args.mode = img_mode

    if args.clip == 'video':
        nwucla_testset = hmdb51.HMDB51_Multcrop(data_root="/home/ssd/video_action/hmdb51/rawframes_resize",
                                                clip_len=args.clip_len,
                                                split_path="../data/split_divide/hmdb/test_list1.txt",
                                                mode=img_mode,
                                                sample_interal=1)
    elif args.clip == 'clip':

        nwucla_testset = hmdb51.HMDB51(data_root="/home/ssd/video_action/hmdb51/rawframes_resize",
                                       clip_len=args.clip_len,
                                       split_path="../data/split_divide/hmdb/test_list1.txt",
                                       mode=img_mode,
                                       sample_interal=1)

    testloader = torch.utils.data.DataLoader(nwucla_testset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers)

    net = i3dpt_spp.I3D_SPP(51, modality='rgb')
    net.load_state_dict(torch.load(pretrain_dir)['net'])

    net.cuda()

    if args.parallel:
        net = nn.DataParallel(net)

    with torch.no_grad():
        test(model=net, test_dataloader=testloader)
