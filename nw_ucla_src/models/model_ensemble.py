import torch.nn as nn
from models.resnet_spp import i3_res50_spp


class Ensemble(nn.Module):
    '''
    depth i3d 和rgb i3d 的集成
    不局限数据集和测试协议,输入不同的配对的模型就可以.
    '''

    def __init__(self, rgb_pretrain_dir, depth_pretain_dir, class_num):
        super(Ensemble, self).__init__()
        self.rgb_model = i3_res50_spp(num_classes=class_num, pretrain_dir=rgb_pretrain_dir)
        self.depth_model = i3_res50_spp(num_classes=class_num, pretrain_dir=depth_pretain_dir)

    def forward(self, rgb_batch, depth_batch):
        rgb_pred, patch_score_rgb, rgb_loss_dict = self.rgb_model(rgb_batch)
        depth_pred, patch_score_depth, depth_loss_dict = self.depth_model(depth_batch)

        pred = (rgb_pred + depth_pred) / 2.0
        pred_score = (patch_score_rgb + patch_score_depth) / 2.0
        return pred, pred_score, rgb_loss_dict, depth_loss_dict
