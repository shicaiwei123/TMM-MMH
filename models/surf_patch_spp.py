import torch.nn as nn
import torchvision.models as tm
import torch

from models.resnet18_se import resnet18_se
from lib.model_arch_utils import Flatten, SPP


class SURF_Patch_SPP(nn.Module):
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
        self.avgpool_whole = nn.AdaptiveAvgPool2d((1, 1))
        self.spp = SPP()

    def attention_map(self, fm, eps=1e-6):
        am = torch.pow(torch.abs(fm), 2)
        # am = torch.sum(am, dim=1, keepdim=True)
        norm = torch.norm(am, dim=(2, 3), keepdim=True)
        am = torch.div(am, norm + eps)

        return am

    def forward(self, img_rgb, img_ir, img_depth):
        x_rgb = self.special_bone_rgb(img_rgb)
        x_ir = self.special_bone_ir(img_ir)
        x_depth = self.special_bone_depth(img_depth)
        x = torch.cat((x_rgb, x_ir, x_depth), dim=1)
        x_whole = self.shared_bone(x)

        # patch
        x_layer3 = self.shared_bone[0](x)
        x_layer4 = self.shared_bone[1](x_layer3)

        if self.args.weight_patch:
            x_layer4 = self.attention_map(x_layer4)

        # SPP

        x_feature = self.spp(x_layer4)
        feature_num = x_feature.shape[-1]
        patch_score = torch.zeros(x_feature.shape[0], self.args.class_num, feature_num)
        patch_strength = torch.zeros(x_feature.shape[0], feature_num)
        for i in range(feature_num):
            patch_feature = x_feature[:, :, i]
            patch_feature_flatten = self.shared_bone[3](patch_feature)
            # print(patch_feature_flatten.shape)

            patch_strength[:, i] = torch.mean(patch_feature_flatten, dim=1)
            # print(patch_feature_flatten)
            # print(patch_strength)
            patch_logits = self.shared_bone[4](patch_feature_flatten)
            patch_sigmoid = torch.sigmoid(patch_logits[:, 0] - patch_logits[:, 1])
            patch_score[:, :, i] = patch_logits

        return x_whole, patch_score, patch_strength
