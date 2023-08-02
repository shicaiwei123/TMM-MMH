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
from loss.focalloss import FocalLoss
import torch.nn as nn


class CB_loss(nn.Module):
    def __init__(self, samples_per_cls, class_num, loss_type, beta=0.99, gamma=2.0):
        super(CB_loss, self).__init__()
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * class_num

        self.weights = weights
        self.loss_type = loss_type
        self.gamma = gamma
        self.class_num = class_num

    def forward(self, inputs, targets):

        labels_one_hot = F.one_hot(targets, self.class_num).float()
        labels_one_hot = labels_one_hot.cpu()

        weights = torch.tensor(self.weights).float()
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, self.class_num)

        weights = weights.cuda()

        if self.loss_type == "focal":
            focal_loss = FocalLoss(class_num=self.class_num, alpha=weights, gamma=self.gamma)
            cb_loss = focal_loss(inputs, targets)
        elif self.loss_type == "sigmoid":
            cb_loss = F.binary_cross_entropy_with_logits(input=inputs, target=self.labels_one_hot, weights=weights)
        else:
            pred = inputs.softmax(dim=1)
            cb_loss = F.binary_cross_entropy(input=pred, target=self.labels_one_hot, weight=weights)
        return cb_loss


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
