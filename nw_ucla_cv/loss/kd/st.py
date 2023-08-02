from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftTarget(nn.Module):
    '''
    Distilling the Knowledge in a Neural Network
    https://arxiv.org/pdf/1503.02531.pdf
    '''

    def __init__(self, T):
        super(SoftTarget, self).__init__()
        self.T = T

    def forward(self, out_s, out_t):
        loss = F.kl_div(F.log_softmax(out_s / self.T, dim=1),
                        F.softmax(out_t / self.T, dim=1),
                        reduction='batchmean') * self.T * self.T

        return loss


class MultiSoftTarget(nn.Module):
    def __init__(self, T):
        super(MultiSoftTarget, self).__init__()
        self.T = T

    def forward(self, out_s_multi, out_t_multi):
        loss_sum = torch.tensor(0.0)

        # print(out_s_multi.shape)
        # print(out_s_multi[0, 1])
        for i in range(out_s_multi.shape[1]):
            out_s = torch.squeeze(out_s_multi[:, i, :], dim=1)
            out_t = torch.squeeze(out_t_multi[:, i, :], dim=1)
            # print(out_s,out_t)
            # print(out_t.shape)

            loss = F.kl_div(F.log_softmax(out_s / self.T, dim=1),
                            F.softmax(out_t / self.T, dim=1),
                            reduction='batchmean') * self.T * self.T
            # print(loss)
            loss_sum = loss_sum + loss

        return loss_sum/out_s_multi.shape[1]
