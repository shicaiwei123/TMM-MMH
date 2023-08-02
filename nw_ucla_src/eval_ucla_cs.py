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
parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training')
parser.add_argument('--parallel', action='store_true', default=False)
parser.add_argument('--workers', type=int, default=2)
parser.add_argument('--mode', default='clip', help='video|clip')
parser.add_argument('--model', default='r50_nl', help='r50|r50_nl')
args = parser.parse_args()


# suject 10
# RGB | clip 88.7 | video |0.925
# depth | clip 88.3|video | 0.933    |clip+3crop 83.3

# spp clip 933    video 950   model  pretrain_dir = "/home/icml//shicaiwei/multi_model_fas/output/models/multiKD/_ckpt_E_200_I_0.pth"
# cdd clip 917/908    video 0.942/0.925       model  CDD/_ckpt_E_29_I_2.pth"   CDD/_ckpt_E_27_I_2.pth"

# 好像是有些高,又担心其他都是

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
    # pretrain_dir = "/home/icml//shicaiwei/multi_model_fas/output/models/multiKD/_ckpt_E_290_I_0.pth"
    # pretrain_dir = "/home/icml/shicaiwei/multi_model_fas/output/models/ckpt_E_224_I_1.pth"
    pretrain_dir = "/home/icml//shicaiwei/multi_model_fas/output/models/CDD/_ckpt_E_27_I_2.pth"

    args.split_subject = 10

    args.clip_len = 16

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

    net = resnet.i3_res50(10, pretrain_dir=pretrain_dir)

    net.cuda()

    if args.parallel:
        net = nn.DataParallel(net)

    with torch.no_grad():
        test(model=net, test_dataloader=testloader)
