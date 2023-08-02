import torch.nn as nn
import torchvision.models as tm
import torch

from models.resnet18_se import resnet18_se
from lib.model_arch_utils import Flatten
from lib.model_arch import ROI_Pooling


class SURF_SIFT_Patch(nn.Module):
    def __init__(self, args):
        super().__init__()

        model_resnet18_se_1 = resnet18_se(args, pretrained=False)
        model_resnet18_se_2 = resnet18_se(args, pretrained=False)
        model_resnet18_se_3 = resnet18_se(args, pretrained=False)

        self.special_bone_rgb = nn.Sequential(model_resnet18_se_1.conv1,
                                              model_resnet18_se_1.bn1,
                                              model_resnet18_se_1.relu,
                                              model_resnet18_se_1.maxpool,
                                              model_resnet18_se_1.layer1,
                                              model_resnet18_se_1.layer2,
                                              model_resnet18_se_1.se_layer)
        self.special_bone_ir = nn.Sequential(model_resnet18_se_2.conv1,
                                             model_resnet18_se_2.bn1,
                                             model_resnet18_se_2.relu,
                                             model_resnet18_se_2.maxpool,
                                             model_resnet18_se_2.layer1,
                                             model_resnet18_se_2.layer2,
                                             model_resnet18_se_2.se_layer)
        self.special_bone_depth = nn.Sequential(model_resnet18_se_3.conv1,
                                                model_resnet18_se_2.bn1,
                                                model_resnet18_se_2.relu,
                                                model_resnet18_se_2.maxpool,
                                                model_resnet18_se_2.layer1,
                                                model_resnet18_se_2.layer2,
                                                model_resnet18_se_2.se_layer)

        self.shared_bone = nn.Sequential(model_resnet18_se_1.layer3_new,
                                         model_resnet18_se_1.layer4,
                                         nn.AdaptiveAvgPool2d((1, 1)),
                                         Flatten(1),
                                         model_resnet18_se_1.fc,
                                         model_resnet18_se_1.dropout,
                                         )
        self.args = args
        self.roi_pooling = ROI_Pooling()

    def forward(self, img_rgb, img_ir, img_depth,cluster_center):
        x_rgb = self.special_bone_rgb(img_rgb)
        x_ir = self.special_bone_ir(img_ir)
        x_depth = self.special_bone_depth(img_depth)
        x = torch.cat((x_rgb, x_ir, x_depth), dim=1)
        x_whole = self.shared_bone(x)

        # pacth
        x_layer3 = self.shared_bone[0](x)
        x_layer4 = self.shared_bone[1](x_layer3)
        feature_list = self.roi_pooling(x_layer4, cluster_center, spatial_ratio=28)
        patch_score = torch.zeros((x.shape[0], 1))

        for i in range(len(feature_list)):
            patch_feature_flatten = feature_list[i]
            patch_logits = self.shared_bone[4](patch_feature_flatten)
            patch_sigmoid = torch.sigmoid(patch_logits[:, 0] - patch_logits[:, 1])
            patch_score[:, i] = patch_sigmoid

        return x_whole, patch_score
