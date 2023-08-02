import sys

sys.path.append('..')
from models.resnet_se_spp import resnet18_se_patch_spp
from src.surf_baseline_multi_dataloader import surf_multi_transforms_train, surf_multi_transforms_test
from lib.model_develop import calc_accuracy_kd_patch
from datasets.surf_txt import SURF, SURF_generate
from configuration.config_patch_kd import args
import torch
import torch.nn as nn
import os


def batch_test(model, args):
    '''
    利用dataloader 装载测试数据,批次进行测试
    :return:
    '''

    root_dir = "/home/bigspace/shicaiwei/liveness_data/CASIA-SURF"
    txt_dir = root_dir + '/test_private_list.txt'

    surf_dataset = SURF(txt_dir=txt_dir,
                        root_dir=root_dir,
                        transform=surf_multi_transforms_test, miss_modal=args.miss_modal)
    #
    # surf_dataset = SURF_generate(rgb_dir=args.rgb_root, depth_dir=args.depth_root, ir_dir=args.ir_root,
    #                              transform=surf_multi_transforms_test)

    test_loader = torch.utils.data.DataLoader(
        dataset=surf_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4
    )

    result = calc_accuracy_kd_patch(model=model, loader=test_loader, args=args, verbose=True, hter=True)
    print(result)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(1)

    for i in range(4):
        # print(i)
        i = i + 1

        # pretrain_dir = "../output/models/surf_patch_kd_multikd_avg_self_attention_21_multi_multi_rgb_lr_0.001_version_" + str(
        #     i) + "_lambda_kd_0.8_weight_patch_1_hter_best_.pth"

        pretrain_dir = "../output/models/surf_patch_kd_multikd_avg_attention_adam_multi_multi_rgb_lr_0.001_version_2_lambda_kd_0.4_weight_patch_1_hter_best_.pth"
        args.gpu = 1

        args.miss_modal = 0
        args.backbone = "resnet18_se"
        args.modal = 'rgb'
        args.student_data = 'multi_rgb'
        args.weight_patch = pretrain_dir.split('_')[-1].split('.')[0]
        model = resnet18_se_patch_spp(args)
        model.load_state_dict(torch.load(pretrain_dir))

        batch_test(model=model, args=args)
