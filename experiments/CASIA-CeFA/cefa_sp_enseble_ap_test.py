import sys

sys.path.append('..')
from models.sp_ap_ensemble import SAP
from src.cefa_baseline_multi_dataloader import cefa_multi_transforms_train, cefa_multi_transforms_test
from lib.model_develop import calc_accuracy_ensemble
from datasets.cefa_multi_protocol import CEFA_Multi
from configuration.config_sp_ap_ensemble import args
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

    result = calc_accuracy_ensemble(model=model, loader=test_loader, verbose=True, hter=True)
    print(result)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(1)

    # sp_ensemble_ap_multiKD_acer_best__lr_0.0001_version_2_acer_best_.pth

    for i in range(3):
        try:
            i = i + 0
            pretrain_dir = "../output/models/cefa_sp_ensemble_ap_multiKD_acer_best__lr_0.0001_version_0_acer_best_.pth"

            args.backbone = "resnet18_se"
            model = SAP(args, "../output/models/cefa_patch_feature_l3_l4_multi_multi_rgb_lr_0.001_version_0_sift_0_acer_best_.pth",
                        "../output/models/cefa_patch_kd_multiKD_new_multi_multi_rgb_lr_0.001_version_0_weight_patch_0_select_0_acer_best_.pth", )
            # test_para = torch.load(pretrain_dir)
            # model.load_state_dict(torch.load(pretrain_dir))

            batch_test(model=model, args=args)
        except Exception as e:
            print(e)
            continue
