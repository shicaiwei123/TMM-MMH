import sys

sys.path.append('..')
import torch.nn as nn
import torch
import torchvision.models as tm
from models.resnet_se_patch_feature import resnet18_se_patch_feature
from models.resnet_se_spp import resnet18_se_patch_spp


class SAP(nn.Module):
    def __init__(self, args, sp_pretrained_dir=None, ap_pretrained_dir=None):
        super(SAP, self).__init__()
        self.sp_pretrained = sp_pretrained_dir
        self.ap_pretrained = ap_pretrained_dir
        self.sp_model = resnet18_se_patch_feature(args=args)
        if sp_pretrained_dir is not None:
            para_dict = torch.load(sp_pretrained_dir)
            init_para_dict = self.sp_model.state_dict()
            try:
                for k, v in para_dict.items():
                    if k in init_para_dict:
                        init_para_dict[k] = para_dict[k]
                self.sp_model.load_state_dict(init_para_dict)
            except Exception as e:
                print(e)

        if args.sp_freeze:
            for p in self.sp_model.parameters():
                p.requires_grad = False

        self.sp_model.eval()

        self.ap_model = resnet18_se_patch_spp(args=args)
        if ap_pretrained_dir is not None:
            para_dict = torch.load(ap_pretrained_dir)
            init_para_dict = self.ap_model.state_dict()
            try:
                for k, v in para_dict.items():
                    if k in init_para_dict:
                        init_para_dict[k] = para_dict[k]
                self.ap_model.load_state_dict(init_para_dict)
            except Exception as e:
                print(e)

        if args.ap_freeze:
            for p in self.ap_model.parameters():
                p.requires_grad = False

        self.ap_model.eval()

        self.fc = nn.Linear(4, 2, bias=True)
        self.fuse_weight_1 = torch.nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.fuse_weight_2 = torch.nn.Parameter(torch.Tensor(1), requires_grad=True)

        self.fuse_weight_1.data.fill_(0.5)
        self.fuse_weight_2.data.fill_(0.5)

    def forward(self, x):
        x_sp = self.sp_model(x)[0]
        x_ap = self.ap_model(x)[0]

        # x_contact = torch.cat((x_ap, x_sp), dim=1)
        # x_out = x_ap * self.fuse_weight_2 + x_sp * self.fuse_weight_1
        x_out = x_ap  + x_sp
        return x_out
