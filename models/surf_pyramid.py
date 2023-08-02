import torch.nn as nn
import torchvision.models as tm
import torch

from models.resnet18_se import resnet18_se
from lib.model_arch_utils import Flatten


class SURF_Pyramid(nn.Module):
    def __init__(self, args, pretrained=False):
        super().__init__()

        model_resnet18_se_1 = resnet18_se(args, pretrained=pretrained)

        self.bone = nn.Sequential(model_resnet18_se_1.conv1,
                                  model_resnet18_se_1.bn1,
                                  model_resnet18_se_1.relu,
                                  # model_resnet18_se.maxpool,
                                  model_resnet18_se_1.layer1,
                                  model_resnet18_se_1.layer2,
                                  model_resnet18_se_1.se_layer,
                                  model_resnet18_se_1.layer3,
                                  model_resnet18_se_1.layer4,
                                  )

        self.args = args
        # # depth
        # self.pool_32x32_depth = nn.AdaptiveAvgPool2d((32, 32))
        # self.conv_1x1_32_depth = nn.Conv2d(512, 1, 1)
        # self.pool_16x16_depth = nn.AdaptiveAvgPool2d((16, 16))
        # self.conv_1x1_16_depth = nn.Conv2d(512, 1, 1)
        #
        # # ir
        # self.pool_32x32_ir = nn.AdaptiveAvgPool2d((32, 32))
        # self.conv_1x1_32_ir = nn.Conv2d(512, 1, 1)
        # self.pool_16x16_ir = nn.AdaptiveAvgPool2d((16, 16))
        # self.conv_1x1_16_ir = nn.Conv2d(512, 1, 1)

        self.decoder_ir_32x32 = nn.Sequential(

            nn.Conv2d(512, 1, kernel_size=1, padding=0),
            nn.AdaptiveAvgPool2d((32, 32)),
            # nn.Sigmoid()
        )
        self.decoder_depth_32x32 = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1, padding=0),
            nn.AdaptiveAvgPool2d((32, 32)),
            # nn.Sigmoid()
        )

        self.decoder_ir_16x16 = nn.Sequential(

            nn.Conv2d(512, 1, kernel_size=1, padding=0),
            nn.AdaptiveAvgPool2d((16, 16)),
            # nn.Sigmoid()
        )
        self.decoder_depth_16x16 = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1, padding=0),
            nn.AdaptiveAvgPool2d((16, 16)),
            # nn.Sigmoid()
        )

        # binary
        self.pool_8x8 = nn.AdaptiveAvgPool2d((8, 8))
        self.conv_1x1_8 = nn.Conv2d(512, 1, 1)
        self.pool_4x4 = nn.AdaptiveAvgPool2d((4, 4))
        self.conv_1x1_4 = nn.Conv2d(512, 1, 1)
        self.pool_2x2 = nn.AdaptiveAvgPool2d((2, 2))
        self.conv_1x1_2 = nn.Conv2d(512, 1, 1)
        self.pool_1x1 = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_1x1_1 = nn.Conv2d(512, 1, 1)

        self.fc1 = nn.Linear(64 + 16 + 4 + 1, args.class_num)
        self.fc2 = nn.Linear(2 * (32 * 32 + 16 * 16), args.class_num)
        self.S1 = nn.Sigmoid()

    def forward(self, img_rgb):
        x_rgb = self.bone(img_rgb)

        if self.args.origin_deeppix:
            x_rgb_8x8 = self.pool_8x8(x_rgb)
            x_rgb_8x8_conv = self.conv_1x1_8(x_rgb_8x8)
            x_rgb_4x4 = self.pool_4x4(x_rgb_8x8)
            x_rgb_4x4_conv = self.conv_1x1_4(x_rgb_4x4)
            x_rgb_2x2 = self.pool_2x2(x_rgb_4x4)
            x_rgb_2x2_conv = self.conv_1x1_2(x_rgb_2x2)
            x_rgb_1x1 = self.pool_1x1(x_rgb_2x2)
            x_rgb_1x1_conv = self.conv_1x1_8(x_rgb_1x1)

            x_rgb_8x8_flatten = x_rgb_8x8_conv.view(x_rgb_8x8.shape[0], -1)
            x_rgb_4x4_flatten = x_rgb_4x4_conv.view(x_rgb_4x4.shape[0], -1)
            x_rgb_2x2_flatten = x_rgb_2x2_conv.view(x_rgb_2x2.shape[0], -1)
            x_rgb_1x1_flatten = x_rgb_1x1_conv.view(x_rgb_1x1.shape[0], -1)

            x_rgb_8x8_conv_s = self.S1(x_rgb_8x8_conv)
            x_rgb_4x4_conv_s = self.S1(x_rgb_4x4_conv)
            x_rgb_2x2_conv_s = self.S1(x_rgb_2x2_conv)
            x_rgb_1x1_conv_s = self.S1(x_rgb_1x1_conv)

            x_flatten_all = torch.cat((x_rgb_8x8_flatten, x_rgb_4x4_flatten, x_rgb_2x2_flatten, x_rgb_1x1_flatten),
                                      dim=1)
            out_binary = self.fc1(x_flatten_all)
            out_binary = self.S1(out_binary)
            return x_rgb_8x8_conv_s, x_rgb_4x4_conv_s, x_rgb_2x2_conv_s, x_rgb_1x1_conv_s, out_binary
        else:
            # x_rgb_32x32_depth = self.pool_32x32_depth(x_rgb)
            # x_rgb_32x32_depth_conv = self.conv_1x1_32_depth(x_rgb_32x32_depth)
            # x_rgb_16x16_depth = self.pool_16x16_depth(x_rgb_32x32_depth)
            # x_rgb_16x16_depth_conv = self.conv_1x1_16_depth(x_rgb_16x16_depth)
            #
            # x_rgb_32x32_depth_conv_s = self.S1(x_rgb_32x32_depth_conv)
            # x_rgb_16x16_depth_conv_s = self.S1(x_rgb_16x16_depth_conv)
            #
            # x_rgb_32x32_depth_flatten = x_rgb_32x32_depth_conv.view(x_rgb_32x32_depth.shape[0], -1)
            # x_rgb_16x16_depth_flatten = x_rgb_16x16_depth_conv.view(x_rgb_16x16_depth.shape[0], -1)
            #
            # x_rgb_32x32_ir = self.pool_32x32_ir(x_rgb)
            # x_rgb_32x32_ir_conv = self.conv_1x1_32_ir(x_rgb_32x32_ir)
            # x_rgb_16x16_ir = self.pool_16x16_ir(x_rgb_32x32_ir)
            # x_rgb_16x16_ir_conv = self.conv_1x1_16_ir(x_rgb_16x16_ir)
            #
            # x_rgb_32x32_ir_conv_s = self.S1(x_rgb_32x32_ir_conv)
            # x_rgb_16x16_ir_conv_s = self.S1(x_rgb_16x16_ir_conv)
            #
            # x_rgb_32x32_ir_flatten = x_rgb_32x32_ir_conv.view(x_rgb_32x32_ir.shape[0], -1)
            # x_rgb_16x16_ir_flatten = x_rgb_16x16_ir_conv.view(x_rgb_16x16_ir.shape[0], -1)
            #
            # x_flatten_all = torch.cat(
            #     (x_rgb_32x32_depth_flatten, x_rgb_16x16_depth_flatten, x_rgb_32x32_ir_flatten, x_rgb_16x16_ir_flatten),
            #     dim=1)
            # out_binary = self.fc2(x_flatten_all)
            # out_binary = self.S1(out_binary)
            # return x_rgb_32x32_depth_conv_s, x_rgb_16x16_depth_conv_s, x_rgb_32x32_ir_conv_s, x_rgb_16x16_ir_conv_s, out_binary

            # depth
            x_depth_32x32 = self.decoder_depth_32x32(x_rgb)
            x_depth_16x16 = self.decoder_depth_16x16(x_rgb)
            x_ir_32x32 = self.decoder_ir_32x32(x_rgb)
            x_ir_16x16 = self.decoder_ir_16x16(x_rgb)

            # ir
            x_depth_32x32_s = self.S1(x_depth_32x32)
            x_depth_16x16_s = self.S1(x_depth_16x16)
            x_ir_32x32_s = self.S1(x_ir_32x32)
            x_ir_16x16_s = self.S1(x_ir_16x16)

            # 展开
            x_depth_32x32_s_flatten = x_depth_32x32_s.view(x_depth_32x32_s.shape[0], -1)
            x_depth_16x16_s_flatten = x_depth_16x16_s.view(x_depth_16x16_s.shape[0], -1)
            x_ir_32x32_s_flatten = x_depth_32x32_s.view(x_ir_32x32_s.shape[0], -1)
            x_ir_16x16_s_flatten = x_ir_16x16_s.view(x_ir_16x16_s.shape[0], -1)

            # fc
            x_flatten_all = torch.cat(
                (x_depth_32x32_s_flatten, x_depth_16x16_s_flatten, x_ir_32x32_s_flatten, x_ir_16x16_s_flatten),
                dim=1)
            out_binary = self.fc2(x_flatten_all)
            out_binary = self.S1(out_binary)

            return x_depth_32x32_s, x_depth_16x16_s, x_ir_32x32_s, x_ir_16x16_s, out_binary
