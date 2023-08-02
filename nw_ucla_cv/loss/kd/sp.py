from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F


class SP(nn.Module):
    '''
    Similarity-Preserving Knowledge Distillation
    https://arxiv.org/pdf/1907.09682.pdf
    '''

    def __init__(self):
        super(SP, self).__init__()

    def forward(self, fm_s, fm_t):

        # print(F.mse_loss(fm_s, fm_t))

        fm_s = fm_s.view(fm_s.size(0), -1)

        feat_s_cos_diff = torch.mm(fm_s, fm_s.t())
        feat_s_cos_diff = F.normalize(feat_s_cos_diff, p=2, dim=1)

        fm_t = fm_t.view(fm_t.size(0), -1)
        feat_t_cos_diff = torch.mm(fm_t, fm_t.t())
        feat_t_cos_diff = F.normalize(feat_t_cos_diff, p=2, dim=1)


        feat_t_cos_diff = feat_t_cos_diff.cuda()
        loss = F.mse_loss(feat_s_cos_diff, feat_t_cos_diff)

        return loss
