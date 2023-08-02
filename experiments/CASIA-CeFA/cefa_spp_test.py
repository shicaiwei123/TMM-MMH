import sys

sys.path.append('..')
from models.resnet_se_spp import resnet18_se_patch_spp
from src.cefa_baseline_multi_dataloader import cefa_multi_transforms_train, cefa_multi_transforms_test
from lib.model_develop import calc_accuracy_kd_patch
from datasets.cefa_multi_protocol import CEFA_Multi
from configuration.config_patch_kd_spp import args
import torch
import torch.nn as nn
import os


def batch_test(model, args):
    '''
    利用dataloader 装载测试数据,批次进行测试
    :return:
    '''

    args.data_root = "/home/data/shicaiwei/cefa/CeFA-Race"
    cefa_dataset = CEFA_Multi(args=args, mode='test', protocol=args.protocol, transform=cefa_multi_transforms_test,
                              miss_modal=args.miss_modal)

    test_loader = torch.utils.data.DataLoader(
        dataset=cefa_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4
    )

    result = calc_accuracy_kd_patch(model=model, loader=test_loader, args=args, verbose=True, hter=True)
    print(result)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(1)

    for i in range(8):
        i = i

        # pretrain_dir = "../output/models/patch_kd_multiKD_mmdloss_select_avg_spp_multi_multi_rgb_lr_0.001_version_" + str(
        #     i) + "_weight_patch_1_acer_best_.pth"

        pretrain_dir = "../output/models/surf_patch_kd_multikd_max_multi_multi_rgb_lr_0.001_version_4_weight_patch_0_select_0.pth"

        args.gpu = 1

        args.miss_modal = 0
        args.backbone = "resnet18_se"
        args.modal = 'profile'
        args.student_data = 'multi_rgb'
        args.weight_patch = 0
        model = resnet18_se_patch_spp(args)
        model.load_state_dict(torch.load(pretrain_dir))

        batch_test(model=model, args=args)
