import torch.nn as nn
import torch
import torch.nn.functional as F

from models.resnet_spp import i3_res50_spp
from models.resnet_feature import i3_res50_feature


# add_coefficient = F.softmax(self.fc(spp_cat_cd),dim=0) 0.925| 0.948(video)


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

        self.fc1 = nn.Linear(args.class_num, 2)
        self.fc2 = nn.Linear(args.class_num, 2)
        self.fuse_weight_1 = torch.nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.fuse_weight_2 = torch.nn.Parameter(torch.Tensor(1), requires_grad=True)

        self.fuse_weight_1.data.fill_(0.5)
        self.fuse_weight_2.data.fill_(1)

    def forward(self, rgb_batch):
        spp_pred = self.spp_model(rgb_batch)
        cd_pred = self.cd_model(rgb_batch)

        # print(self.fuse_weight_1,self.fuse_weight_2)
        # print(pred.shape)

        # 注意力呢?
        # 只求两个值---925
        # 那么16个值?,也就是,每个样本都有自己的加权系数.

        # print(spp_pred.is_cuda,cd_pred.is_cuda,pred.is_cuda)
        pred = spp_pred[0] + cd_pred[0] * self.fuse_weight_2

        loss_dict = {}
        if 'label' in rgb_batch:
            loss = F.cross_entropy(pred, rgb_batch['label'], reduction='none')
            loss_dict = {'loss': loss}

        # print(loss_dict)
        return pred, loss_dict

        # # print(spp_pred.shape,cd_pred.shape)
        #
        # spp_pred_bacthmean = torch.mean(spp_pred[0], dim=0, keepdim=True)
        # cd_pred_bacthmean = torch.mean(cd_pred[0], dim=0, keepdim=True)
        #
        # spp_cat_cd = torch.cat((spp_pred_bacthmean, cd_pred_bacthmean), dim=0)
        # # print(spp_pred_bacthmean.shape)
        # a, _ = torch.max(spp_cat_cd, dim=0)
        # # print(a)
        # # print(a.shape)
        #
        # spp_add_cd = (spp_pred_bacthmean + cd_pred_bacthmean) / 2
        #
        # add_coefficient_max = F.sigmoid(self.fc1(a))
        # add_coefficient_mean = F.sigmoid(self.fc2(spp_add_cd))
        #
        # add_coefficient = (add_coefficient_max + add_coefficient_mean) / 2
        # add_coefficient = add_coefficient[0]
        # # print(add_coefficient)
        # # print(add_coefficient.shape)
        #
        # pred = spp_pred[0] * add_coefficient[0] + cd_pred[0] * add_coefficient[1]
