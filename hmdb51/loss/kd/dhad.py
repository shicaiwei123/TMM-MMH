from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F


class LRN3D(nn.Module):
    def __init__(self, input_channel, class_num):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(input_channel, input_channel, 1, 1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv3d(input_channel, input_channel, 1, 1), nn.ReLU())
        self.weight = nn.Sequential(nn.Conv3d(input_channel, input_channel, 1, 1), nn.ReLU(),
                                    nn.AdaptiveMaxPool3d((1, 1, 1)))
        self.mlp = nn.Linear(input_channel, class_num, bias=True)
        self.mse = nn.MSELoss()

    def forward(self, x1, x2):
        F1 = self.conv1(x1)
        F2 = self.conv2(x2)
        weight_out = self.weight(x1)
        weight_out = weight_out.view(weight_out.shape[0], -1)
        mlp_out = self.mlp(weight_out)

        feature_output = self.mse(F1, F2)
        return mlp_out, feature_output


class LRN(nn.Module):
    def __init__(self, input_channel, class_num):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(input_channel, input_channel, 1, 1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(input_channel, input_channel, 1, 1), nn.ReLU())
        self.weight = nn.Sequential(nn.Conv2d(input_channel, input_channel, 1, 1), nn.ReLU(),
                                    nn.AdaptiveMaxPool2d((1, 1)))
        self.mlp = nn.Linear(input_channel, class_num, bias=True)
        self.mse = nn.MSELoss()

    def forward(self, x1, x2):
        F1 = self.conv1(x1)
        F2 = self.conv2(x2)
        weight_out = self.weight(x1)
        weight_out = weight_out.view(weight_out.shape[0], -1)
        mlp_out = self.mlp(weight_out)

        feature_output = self.mse(F1, F2)
        return mlp_out, feature_output


class DHAD3D(nn.Module):
    def __init__(self, input_channel1, input_channel2, input_channel3, class_num):
        super(DHAD3D, self).__init__()
        self.input_dim1 = input_channel1
        self.input_dim1 = input_channel2
        self.input_dim1 = input_channel3
        self.ce = nn.CrossEntropyLoss()

        self.lrn1 = LRN3D(input_channel1, class_num)
        self.lrn2 = LRN3D(input_channel2, class_num)
        self.lrn3 = LRN3D(input_channel3, class_num)

    def forward(self, x1_s, x1_t, x2_s, x2_t, x3_s, x3_t, target):
        mlp_out_1, lri_feature_1 = self.lrn1(x1_s, x1_t)
        mlp_out_2, lri_feature_2 = self.lrn2(x2_s, x2_t)
        mlp_out_3, lri_feature_3 = self.lrn3(x3_s, x3_t)
        loss1 = self.ce(mlp_out_1, target) * lri_feature_1
        loss2 = self.ce(mlp_out_2, target) * lri_feature_2
        loss3 = self.ce(mlp_out_3, target) * lri_feature_3

        loss = loss1 + loss2 + loss3
        return loss

class DHAD(nn.Module):
    def __init__(self, input_channel1, input_channel2, input_channel3, class_num):
        super(DHAD, self).__init__()
        self.input_dim1 = input_channel1
        self.input_dim1 = input_channel2
        self.input_dim1 = input_channel3
        self.ce = nn.CrossEntropyLoss()

        self.lrn1 = LRN(input_channel1, class_num)
        self.lrn2 = LRN(input_channel2, class_num)
        self.lrn3 = LRN(input_channel3, class_num)

    def forward(self, x1_s, x1_t, x2_s, x2_t, x3_s, x3_t, target):
        mlp_out_1, lri_feature_1 = self.lrn1(x1_s, x1_t)
        mlp_out_2, lri_feature_2 = self.lrn2(x2_s, x2_t)
        mlp_out_3, lri_feature_3 = self.lrn3(x3_s, x3_t)
        loss1 = self.ce(mlp_out_1, target) * lri_feature_1
        loss2 = self.ce(mlp_out_2, target) * lri_feature_2
        loss3 = self.ce(mlp_out_3, target) * lri_feature_3

        loss = loss1 + loss2 + loss3
        return loss
