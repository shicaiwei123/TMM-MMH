from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.model_arch_utils import SelfAttention


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


# class MultiSoftTarget(nn.Module):
#     def __init__(self, T):
#         super(MultiSoftTarget, self).__init__()
#         self.T = T
#
#     def forward(self, out_s_multi, out_t_multi):
#         loss_sum = torch.tensor(0.0)
#
#         # print(out_s_multi.shape)
#         # print(out_s_multi[0, 1])
#         for i in range(out_s_multi.shape[2]):
#             out_s = torch.squeeze(out_s_multi[:, :, i], dim=1)
#             out_t = torch.squeeze(out_t_multi[:, :, i], dim=1)
#
#             loss = F.kl_div(F.log_softmax(out_s / self.T, dim=1),
#                             F.softmax(out_t / self.T, dim=1),
#                             reduction='batchmean') * self.T * self.T
#             # print(loss)
#             loss_sum = loss_sum + loss
#
#         return loss_sum/out_s_multi.shape[2]


class LinearWeightedAvg(nn.Module):
    def __init__(self, n_inputs):
        super(LinearWeightedAvg, self).__init__()
        self.weights = nn.ParameterList([nn.Parameter(torch.tensor(1.0) / n_inputs) for i in range(n_inputs)])

    def forward(self, input):
        p_list = []
        for p in self.weights.parameters():
            p_list.append(p)
        print(p_list)
        res = 0
        for emb_idx, emb in enumerate(input):
            res += emb * self.weights[emb_idx]
        return res


class MultiSoftTarget(nn.Module):
    def __init__(self, T):
        super(MultiSoftTarget, self).__init__()
        self.T = T
        self.linear = SelfAttention(21, 21, 21)
        # self.linear = nn.Linear(64, 64, bias=True)

        self.relu1 = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, out_s_multi, out_t_multi):
        loss_sum = torch.tensor(0.0).cuda()
        loss_list = []
        # print(out_s_multi.shape)
        # print(out_s_multi[0, 1])

        # patch_strength = self.relu1(self.linear(patch_strength))
        # patch_strength = torch.permute(patch_strength, (1, 0))

        # print(self.softmax(patch_strength[:, 1]))
        # patch_strength = torch.mean(patch_strength, dim=1)
        # # print(patch_strength)
        # patch_strength = self.softmax(patch_strength)
        # patch_strength = patch_strength.detach()
        # print(patch_strength.shape)
        # patch_strength = torch.from_numpy(patch_strength)
        # print(patch_strength)

        out_s_multi = out_s_multi.cuda()
        out_t_multi = out_t_multi.cuda()

        out_s_multi = out_s_multi.permute(2, 0, 1)
        out_t_multi = out_t_multi.permute(2, 0, 1)

        out_t = torch.reshape(out_t_multi, (out_t_multi.shape[0] * out_t_multi.shape[1], out_t_multi.shape[2]))
        out_s = torch.reshape(out_s_multi, (out_s_multi.shape[0] * out_s_multi.shape[1], out_s_multi.shape[2]))

        # p_s = F.log_softmax(out_s / self.T, dim=1)
        # p_t = F.softmax(out_t / self.T, dim=1)
        # loss_kd = F.kl_div(p_s, p_t, reduction='none') * (self.T ** 2)
        # nan_index = torch.isnan(loss_kd)
        # loss_kd[nan_index] = torch.tensor(0.0).cuda()
        # loss_sum = torch.sum(loss_kd) / (loss_kd.shape[0] - torch.sum(nan_index))

        p_s = F.log_softmax(out_s / self.T, dim=1)
        p_t = F.softmax(out_t / self.T, dim=1)
        loss_kd = F.kl_div(p_s, p_t, reduction='none') * (self.T ** 2)

        loss_kd = torch.sum(loss_kd, dim=1)
        loss_kd = torch.reshape(loss_kd, (out_s_multi.shape[0], out_s_multi.shape[1]))
        loss_kd = torch.mean(loss_kd, dim=1)
        # print(torch.sum(loss_kd) / loss_kd.shape[0])
        # print("loss_kd_shape", loss_kd.shape)
        nan_index = torch.isnan(loss_kd)
        loss_kd[nan_index] = torch.tensor(0.0).cuda()
        #
        loss_sum = torch.sum(self.linear(loss_kd, loss_kd)) / (loss_kd.shape[0] - torch.sum(nan_index))

        # for i in range(out_s_multi.shape[2]):
        #     out_s = torch.squeeze(out_s_multi[:, :, i], dim=1)
        #     out_t = torch.squeeze(out_t_multi[:, :, i], dim=1)
        #
        #     p_s = F.log_softmax(out_s / self.T, dim=1)
        #     p_t = F.softmax(out_t / self.T, dim=1)
        #     loss_kd = F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / out_s.shape[0]
        #     # print(loss_kd.shape)
        #     # loss = F.kl_div(p_s, p_t, size_average=False, reduce=False, reduction='none')
        #     # # print(loss.shape)
        #     # loss_sum_class = torch.sum(loss, dim=1)
        #     # loss_sum_batch = torch.sum(loss_sum_class * patch_strength_soft) * (self.T ** 2)
        #
        #     # print(loss)
        #     # print( patch_strength[i])
        #     loss_sum = loss_sum + loss_kd * 1 / out_s_multi.shape[2]
        #     loss_list.append(loss_kd)
        #
        # loss_list = torch.tensor(loss_list)
        # loss_list = loss_list.cuda()
        # a = self.linear(loss_list)

        # return self.linear(loss_list)
        # print(loss_sum.shape)

        return loss_sum
