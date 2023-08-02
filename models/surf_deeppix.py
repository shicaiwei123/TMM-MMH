import torch.nn as nn
import torchvision.models as tm
import torch

from models.resnet18_se import resnet18_se
from lib.model_arch_utils import Flatten


class SURF_Deeppix(nn.Module):
    def __init__(self, args, pretrained=False):
        super().__init__()

        model_resnet18_se_1 = resnet18_se(args, pretrained=pretrained)
        model_resnet18_se_2 = resnet18_se(args, pretrained=pretrained)

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

        self.decoder_ir = nn.Sequential(

            nn.Conv2d(512, 1, kernel_size=1, padding=0),
            nn.AdaptiveAvgPool2d((32, 32)),
            nn.Sigmoid()
        )
        self.decoder_depth = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1, padding=0),
            nn.AdaptiveAvgPool2d((32, 32)),
            nn.Sigmoid()
        )
        self.fc = nn.Linear(2 * 32 * 32, args.class_num)
        self.S1 = nn.Sigmoid()

    def forward(self, img_rgb):
        x_rgb = self.bone(img_rgb)
        ir_out = self.decoder_ir(x_rgb)
        depth_out = self.decoder_depth(x_rgb)  # depth_out = self.decoder_ir(x_rgb)
        # ir_out_s = self.S1(ir_out)
        # depth_out_s = self.S1(depth_out)

        decoder_contact = torch.cat((ir_out, depth_out), dim=1)
        decoder_contact_vector = decoder_contact.view(decoder_contact.shape[0], -1)
        binary_out = self.fc(decoder_contact_vector)
        binary_out = self.S1(binary_out)

        return ir_out, depth_out, binary_out
