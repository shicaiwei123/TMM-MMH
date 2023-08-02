import torch.nn as nn
import torch
import torchvision.models as tm
import torch.nn.functional as F


class ROI_Pooling(nn.Module):
    '''
    处理单个feature map的 roi 图像信息
    '''

    def __init__(self):
        super().__init__()
        self.avgpool_patch = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool_patch = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, feature_map, cluster_center, spatial_ratio):
        feature_list = []
        cluster_center_mean = torch.mean(cluster_center, dim=0)
        cluster_center_normal = cluster_center_mean / spatial_ratio
        cluster_center_int = torch.floor(cluster_center_normal)
        cluster_center_float = cluster_center_normal - cluster_center_int
        cluster_center_offset = torch.round(cluster_center_float)
        cluster_center_offset = cluster_center_offset * 2 - 1  # 转到[-1,1]
        cluster_center_int = cluster_center_int + 1  # 转到[1,5]
        cluster_center_int = cluster_center_int + cluster_center_offset

        padding = (1, 1, 1, 1)
        # feature_map = F.pad(feature_map, padding, 'constant', 1)

        # for index in range(cluster_center_mean.shape[0]):
        #     coordinate_single = cluster_center_int[index]
        #     coordinate_single=coordinate_single.long()
        #     # x2 是因为python 索引的问题,从0开始,[0:1] 只索引一个
        #
        #     patch = feature_map[:, :,
        #                         coordinate_single[0]:coordinate_single[0] + 2,
        #                         coordinate_single[1]:coordinate_single[1] + 2]
        #
        patch_avg = self.avgpool_patch(feature_map)
        patch_max = self.maxpool_patch(feature_map)
        patch_feature = patch_avg
        patch_flatten = torch.flatten(patch_feature, 1)
        feature_list.append(patch_flatten)

        return feature_list


class SpatialAttention(nn.Module):
    '''
    空间注意力模块
    '''

    def __init__(self, kernel_size=1):
        super(SpatialAttention, self).__init__()

        padding = 0

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        # self.avg = nn.AdaptiveAvgPool2d((3, 3))

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
