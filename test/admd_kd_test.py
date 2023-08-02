import sys

sys.path.append('..')
from models.resnet_se_admd import ADMD
from src.surf_baseline_single_dataloader import surf_single_transforms_test
from datasets.surf_admd_kd import SURF_ADMD_KD
from lib.model_develop import calc_accuracy_kd_admd
from datasets.surf_txt import SURF, SURF_generate
from configuration.config_admd_kd import args
import torch
import torch.nn as nn


def batch_test(model, args):
    '''
    利用dataloader 装载测试数据,批次进行测试
    :return:
    '''

    root_dir = "/home/bbb/shicaiwei/data/liveness_data/CASIA-SURF"
    txt_dir = root_dir + '/test_private_list.txt'
    admd_kd_dataset = SURF_ADMD_KD(txt_dir=txt_dir, root_dir=root_dir,
                                   transform=surf_single_transforms_test)

    test_loader = torch.utils.data.DataLoader(
        dataset=admd_kd_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4
    )

    result = calc_accuracy_kd_admd(model=model, loader=test_loader, args=args, verbose=True, hter=True)
    print(result)


if __name__ == '__main__':
    rgb_pretrained_name = "resnet18_se_dropout_no_seed_rgb.pth"
    depth_pretrained_name = "admd_kd-depth-rgb-val_best-version-2.pth"
    ir_pretrained_name = "admd_kd-ir-rgb-val_best-version-2.pth"
    args.gpu = 2
    args.miss_modal = 0
    args.backbone = "resnet18_se"
    args.modal = 'rgb'
    args.student_data = 'multi_rgb'
    model = ADMD(args, rgb_name=rgb_pretrained_name, depth_name=depth_pretrained_name, ir_name=ir_pretrained_name)

    batch_test(model=model, args=args)
