import sys

sys.path.append('..')
from models.resnet18_se import resnet18_se
from src.surf_baseline_multi_dataloader import surf_multi_transforms_train, surf_multi_transforms_test
from lib.model_develop import calc_accuracy_kd
from datasets.surf_txt import SURF, SURF_generate
from configuration.config_multi_rgb_kd import args
import torch
import torch.nn as nn


def batch_test(model, args):
    '''
    利用dataloader 装载测试数据,批次进行测试
    :return:
    '''

    root_dir = "/home/bbb/shicaiwei/data/liveness_data/CASIA-SURF"
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

    result = calc_accuracy_kd(model=model, loader=test_loader, args=args, verbose=True, hter=True)
    print(result)


if __name__ == '__main__':
    pretrain_dir = "../output/models/multi_rgb_kd_random_init_multi_multi_rgb_lr_0.001_logits_kd_0_feature_kd_1_version_0.pth"
    args.gpu = 2
    args.miss_modal = 0
    args.backbone = "resnet18_se"
    args.modal = 'rgb'
    args.student_data = 'multi_rgb'
    model = resnet18_se(args)
    model.load_state_dict(torch.load(pretrain_dir))

    batch_test(model=model, args=args)
