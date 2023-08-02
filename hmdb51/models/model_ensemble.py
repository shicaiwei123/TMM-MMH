import torch.nn as nn
import torch
from models import i3dpt_spp
import torch.nn.functional as F


class Ensemble_I3D(nn.Module):
    '''
    flow i3d 和rgb i3d 的集成
    不局限数据集和测试协议,输入不同的配对的模型就可以.
    '''

    def __init__(self, rgb_pretrain_dir, flow_pretain_dir, class_num):
        super(Ensemble_I3D, self).__init__()
        self.rgb_model = i3dpt_spp.I3D_SPP(num_classes=class_num, modality='rgb')
        self.rgb_model.load_state_dict(torch.load(rgb_pretrain_dir)['net'])

        self.flow_model = i3dpt_spp.I3D_SPP(num_classes=class_num, modality='flow')
        self.flow_model.load_state_dict(torch.load(flow_pretain_dir)['net'])

    def forward(self, rgb_batch, flow_batch):
        rgb_pred, patch_score_rgb, rgb_loss_dict = self.rgb_model(rgb_batch)
        flow_pred, patch_score_flow, flow_loss_dict = self.flow_model(flow_batch)

        pred = rgb_pred * 0.5 + flow_pred * 0.5
        patch_score = patch_score_rgb * 0.5 + patch_score_flow * 0.5
        return pred, patch_score, rgb_loss_dict, flow_loss_dict


class Ensemble_I3D_RGB(nn.Module):
    '''
    flow i3d 和rgb i3d 的集成
    不局限数据集和测试协议,输入不同的配对的模型就可以.
    '''

    def __init__(self, rgb_pretrain_path, transfer_pretain_dir, class_num):
        super(Ensemble_I3D_RGB, self).__init__()
        self.rgb_model = i3dpt_spp.I3D_SPP(num_classes=class_num, modality='rgb')
        self.rgb_model.load_state_dict(torch.load(rgb_pretrain_path)['net'])

        self.flow_model = i3dpt_spp.I3D_SPP(num_classes=class_num, modality='rgb')
        self.flow_model.load_state_dict(torch.load(transfer_pretain_dir)['net'])

        for p in self.rgb_model.parameters():
             p.requires_grad = False
        for p in self.flow_model.parameters():
            p.requires_grad = False

        for p in self.rgb_model.conv3d_0c_1x1.parameters():
            p.requires_grad = True
        for p in self.flow_model.conv3d_0c_1x1.parameters():
            p.requires_grad = True


    def forward(self, rgb_batch):
        rgb_pred, patch_score_rgb, rgb_loss_dict = self.rgb_model(rgb_batch)
        flow_pred, patch_score_flow, flow_loss_dict = self.flow_model(rgb_batch)

        pred = rgb_pred * 0.5 + flow_pred * 0.5

        if 'label' in rgb_batch:
            loss = F.cross_entropy(pred, rgb_batch['label'], reduction='none')
            loss_dict = {'loss': loss}

        return pred, loss_dict
