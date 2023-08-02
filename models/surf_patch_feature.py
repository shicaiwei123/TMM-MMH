import torch.nn as nn
import torchvision.models as tm
import torch

from models.resnet18_se import resnet18_se
from lib.model_arch_utils import Flatten


class SURF_Patch_Feature(nn.Module):
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
        self.avgpool_patch = nn.AdaptiveAvgPool2d((args.patch_num, args.patch_num))
        self.maxpool_patch = nn.AdaptiveMaxPool2d((args.patch_num, args.patch_num))
        self.avgpool_whole = nn.AdaptiveAvgPool2d((1, 1))

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
        x_patch_avg = self.avgpool_patch(x_layer4)
        x_patch_max = self.maxpool_patch(x_layer4)

        if self.args.sift:
            # a_avg = x_patch_avg
            a_avg = x_patch_avg + x_patch_max

            coordinates = torch.tensor([[0, 0], [0, 1], [0, 2], [1, 0], [2, 0], [1, 1], [1, 2], [2, 1], [2, 2]])
            patch_score = torch.zeros((a_avg.shape[0], coordinates.shape[0]))

            for i in range(coordinates.shape[0]):
                coordinate = coordinates[i]
                patch_feature = a_avg[:, :, coordinate[0]:coordinate[0] + 2, coordinate[1]:coordinate[1] + 2]
                patch_feature = self.avgpool_whole(patch_feature)
                patch_feature_flatten = torch.flatten(patch_feature, 1)
                patch_logits = self.shared_bone[4](patch_feature_flatten)
                patch_sigmoid = torch.sigmoid(patch_logits[:, 0] - patch_logits[:, 1])
                patch_score[:, i] = patch_sigmoid

        else:
            a_avg = x_patch_avg

            # a_avg = self.attention_map(a_avg)

            patch_num = a_avg.shape[2] * a_avg.shape[3]
            a_avg = torch.reshape(a_avg, (a_avg.shape[0], a_avg.shape[1], patch_num))
            patch_score = torch.zeros((a_avg.shape[0], patch_num))

            for i in range(patch_num):
                patch_feature = a_avg[:, :, i]
                patch_feature_flatten = self.shared_bone[3](patch_feature)
                patch_logits = self.shared_bone[4](patch_feature_flatten)
                patch_sigmoid = torch.sigmoid(patch_logits[:, 0] - patch_logits[:, 1])
                patch_score[:, i] = patch_sigmoid

        return x_whole, patch_score, x_layer3, x_layer4
