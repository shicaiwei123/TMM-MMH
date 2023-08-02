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

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=8, type=int, help='Batch size for training')
parser.add_argument('--parallel', action='store_true', default=False)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--mode', default='clip', help='video|clip')
parser.add_argument('--model', default='r50_nl', help='r50|r50_nl')
args = parser.parse_args()



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
import ntud60
from models import resnet

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    # pretrain_dir = "/home/icml//shicaiwei/multi_model_fas/output/models/multiKD/_ckpt_E_290_I_0.pth"
    # pretrain_dir = "/home/icml/shicaiwei/multi_model_fas/output/models/ckpt_E_224_I_1.pth"
    # pretrain_dir = "/home/CVPR/shicaiwei/multi_model_fas/output/models/multiKD_lambda_kd/_ckpt_E_68_I_1.pth"
    pretrain_dir = "/home/CVPR/shicaiwei/multi_model_fas/output/models/CDD_ntud60_backup/_ckpt_E_70_I_2.pth"
    # pretrain_dir = "/home/CVPR/shicaiwei/multi_model_fas/output/models/multiKD_lambda_kd_new/_ckpt_E_142_I_2.pth"


    args.clip_len = 16
    args.data_root="/home/data/NTUD60"

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

    net = resnet.i3_res50(60, pretrain_dir=pretrain_dir)

    net.cuda()

    if args.parallel:
        net = nn.DataParallel(net)

    with torch.no_grad():
        test(model=net, test_dataloader=testloader)
