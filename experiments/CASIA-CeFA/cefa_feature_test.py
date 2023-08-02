import sys

sys.path.append('..')
from models.resnet_se_patch_feature import resnet18_se_patch_feature
from src.cefa_baseline_multi_dataloader import cefa_multi_transforms_train, cefa_multi_transforms_test
from lib.model_develop import calc_accuracy_kd_patch_feature, calc_accuracy_multi
from datasets.cefa_multi_protocol import CEFA_Multi
from configuration.config_patch_feature_kd import args
import torch
import torch.nn as nn


def batch_test(model, args):
    '''
    利用dataloader 装载测试数据,批次进行测试
    :return:
    '''
    args.data_root = "/home/data/shicaiwei/cefa/CeFA-Race"
    cefa_dataset = CEFA_Multi(args=args, mode='test', miss_modal=args.miss_modal, protocol=args.protocol,
                              transform=cefa_multi_transforms_test)
    cefa_data_loader = torch.utils.data.DataLoader(
        dataset=cefa_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4)

    result = calc_accuracy_kd_patch_feature(args=args,model=model, loader=cefa_data_loader, verbose=True, hter=True)
    print(result)


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(1)

    pretrain_dir = "../output/models/surf_patch_feature_l3_l4_at_multi_multi_rgb_lr_0.001_version_0_sift_0.pth"
    args.gpu = 3
    args.miss_modal = 0
    args.backbone = "resnet18_se"
    args.modal = 'profile'
    args.student_data = 'multi_rgb'
    args.sift = False
    model = resnet18_se_patch_feature(args)
    model.load_state_dict(torch.load(pretrain_dir))

    batch_test(model=model, args=args)
