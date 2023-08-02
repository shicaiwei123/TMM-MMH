import sys

sys.path.append('..')
from models.resnet18_se import resnet18_se
from src.surf_baseline_single_dataloader import surf_single_transforms_test_cycle
from lib.model_develop_utils import calc_accuracy_ensemble
from datasets.surf_single_txt import SURF_Single
from configuration.config_baseline_single import args
import torch
import torch.nn as nn
import torchvision.models as tm


def batch_test(rgb_model, depth_model, args):
    '''
    利用dataloader 装载测试数据,批次进行测试
    :return:
    '''

    root_dir = "/home/bbb/shicaiwei/data/liveness_data/CASIA-SURF"
    txt_dir = root_dir + '/test_private_list.txt'
    surf_dataset = SURF_Single(txt_dir=txt_dir,
                               root_dir=root_dir,
                               transform=surf_single_transforms_test_cycle, modal=args.modal)

    test_loader = torch.utils.data.DataLoader(
        dataset=surf_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
    )

    result = calc_accuracy_ensemble(rgb_model=rgb_model, depth_model=depth_model, loader=test_loader, verbose=True,
                                    hter=True)
    print(result)


if __name__ == '__main__':
    pretrain_dir = "../output/models/resnet18_se_dropout_no_seed_cycle_ir_version_1.pth"
    args.modal = 'ir'
    args.gpu = 3
    args.data_root = ""
    args.backbone = "resnet18_se"
    args.version = 0
    rgb_model = resnet18_se(args, pretrained=False)

    pretrain_dir = "../output/models/resnet18_se_dropout_no_seed_rgb_version_1.pth"
    args.modal = 'ir'
    args.gpu = 3
    args.data_root = ""
    args.backbone = "resnet18_se"
    args.version = 0
    depth_model = resnet18_se(args, pretrained=False)

    depth_model.load_state_dict(torch.load(pretrain_dir))

    batch_test(rgb_model=rgb_model, depth_model=depth_model, args=args)
