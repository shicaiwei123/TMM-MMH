"""Pytorch implementation of Class-Balanced-Loss
   Reference: "Class-Balanced Loss Based on Effective Number of Samples"
   Authors: Yin Cui and
               Menglin Jia and
               Tsung Yi Lin and
               Yang Song and
               Serge J. Belongie
   https://arxiv.org/abs/1901.05555, CVPR'19.
"""

import numpy as np
import torch
import torch.nn.functional as F
import random
from loss.bce_balanceed_loss import BCE_balance_Loss
import torch.nn as nn


class CBBCEB_loss(nn.Module):
    def __init__(self, samples_per_cls, class_num, beta=0.99, gamma=2.0, size_average=None):
        super(CBBCEB_loss, self).__init__()

        self.samples_per_cls = samples_per_cls
        self.gamma = gamma
        self.class_num = class_num
        self.beta = beta
        self.size_average = size_average

    def forward(self, inputs, targets):

        effective_num = 1.0 - np.power(self.beta, self.samples_per_cls)
        weights = (1.0 - self.beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * self.class_num

        labels_one_hot = F.one_hot(targets, self.class_num).float()

        weights = torch.tensor(weights).float().cuda()
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, self.class_num)

        # 对输出节点进行加权
        bceb_loss = BCE_balance_Loss(class_num=self.class_num, alpha=self.samples_per_cls, beta=self.beta)

        # 对batch 进行加权
        bceb_loss_batch = bceb_loss(inputs, targets)
        bceb_loss_weight = weights * bceb_loss_batch

        if self.size_average is None:
            return bceb_loss_weight
        else:
            if self.size_average:
                loss = bceb_loss_weight.mean()
            else:
                loss = bceb_loss_weight.sum()
            return loss


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == '__main__':
    seed_torch(2)
    no_of_classes = 5
    logits = torch.rand(10, no_of_classes).float()
    labels = torch.tensor([0, 1, 2, 2, 2, 2, 2, 3, 4, 4])
    beta = 0.9999
    gamma = 2.0
    # samples_per_cls = [1, 2, 2, 13, 2]
    samples_per_cls = [1, 1, 6, 1, 3]
    loss_type = "focal"
    cb_loss = CB_loss(samples_per_cls, no_of_classes, loss_type, beta, gamma)
    cb = cb_loss(logits, labels)
    print(cb)
