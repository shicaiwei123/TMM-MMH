import sys

sys.path.append('..')
from models.resnet_se_patch import resnet18_se_patch
from models.resnet18_se import resnet18_se
from models.resnet_se_patch_attention import resnet18_se_patch_attention
from src.surf_baseline_multi_dataloader import surf_multi_transforms_train, surf_multi_transforms_test
from src.surf_baseline_single_dataloader import surf_single_transforms_test
from lib.model_develop import calc_accuracy_kd_patch,calc_accuracy
from datasets.casia_fasd import CASIA_Dataset, CASIA_Single
from configuration.config_casia_kd import args
import torch
import torch.nn as nn
import os


def batch_test(model, args):
    '''
    利用dataloader 装载测试数据,批次进行测试
    :return:
    '''

    living_test_dir = os.path.join(args.test_dir, 'living')
    spoofing_test_dir = os.path.join(args.test_dir, 'spoofing')

    casia_dataset = CASIA_Single(living_dir=living_test_dir, spoofing_dir=spoofing_test_dir, args=args,
                                  data_transform=surf_single_transforms_test, balance=False,
                                  sampe_interal=args.sample_interal)

    test_loader = torch.utils.data.DataLoader(
        dataset=casia_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4
    )

    result = calc_accuracy(model=model, loader=test_loader, verbose=True, hter=True)
    print(result)


if __name__ == '__main__':
    pretrain_dir = "../output/models/patch_kd_no_maxpool_multi_multi_rgb_lr_0.001_version_3.pth"
    args.gpu = 3
    args.miss_modal = 0
    args.backbone = "resnet18_se"
    args.modal = 'rgb'
    args.student_data = 'multi_rgb'
    args.weight_patch = False
    args.sift = False
    model = resnet18_se(args)
    model.load_state_dict(torch.load(pretrain_dir))

    batch_test(model=model, args=args)
