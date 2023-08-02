import torch.nn as nn
import torch
import torch.nn.functional as F

from models.resnet_spp import i3_res50_spp
from models.resnet_feature import i3_res50_feature


class Ensemble_SPP_CD(nn.Module):
    '''
    depth i3d 和rgb i3d 的集成
    不局限数据集和测试协议,输入不同的配对的模型就可以.
    '''

    def __init__(self, args, spp_pretrained_dir, cp_pretrained_dir):
        super(Ensemble_SPP_CD, self).__init__()

        self.spp_model = i3_res50_spp(num_classes=args.class_num, pretrain_dir=spp_pretrained_dir)
        if args.spp_freeze:
            for p in self.spp_model.parameters():
                p.requires_grad = False
        self.spp_model.eval()

        self.cd_model = i3_res50_feature(num_classes=args.class_num, pretrain_dir=cp_pretrained_dir)
        if args.cd_freeze:
            for p in self.cd_model.parameters():
                p.requires_grad = False

        self.cd_model.eval()

        self.fc = nn.Linear(2 * args.class_num, args.class_num)

    def forward(self, rgb_batch):
        spp_pred = self.spp_model(rgb_batch)
        cd_pred = self.cd_model(rgb_batch)

        # print(spp_pred.shape,cd_pred.shape)


        spp_pred = spp_pred[0]
        cd_pred = cd_pred[0]


        spp_cat_cd = torch.cat((spp_pred, cd_pred), dim=1)

        pred = self.fc(spp_cat_cd)

        # print(spp_pred.is_cuda,cd_pred.is_cuda,pred.is_cuda)

        loss_dict = {}
        if 'label' in rgb_batch:
            loss = F.cross_entropy(pred, rgb_batch['label'], reduction='none')
            loss_dict = {'loss': loss}

        # print(loss_dict)
        return pred, loss_dict
