import sys

sys.path.append('..')
from models.resnet18_se import resnet18_se
from src.surf_deeppix_dataloader import surf_deeppix_transforms_test
from lib.model_develop_utils import calc_accuracy_multi_advisor
from datasets.surf_multi_advisor import SURF_Multi_Advisor
from models.surf_deeppix import SURF_Deeppix
from configuration.config_deeppix_multi import args
import torch
import torch.nn as nn


def batch_test(model, args):
    '''
    利用dataloader 装载测试数据,批次进行测试
    :return:
    '''

    root_dir = "/home/bbb/shicaiwei/data/liveness_data/CASIA-SURF"
    txt_dir = root_dir + '/test_private_list.txt'
    surf_dataset = SURF_Multi_Advisor(txt_dir=txt_dir,
                                      root_dir=root_dir,
                                      transform=surf_deeppix_transforms_test, args=args)

    test_loader = torch.utils.data.DataLoader(
        dataset=surf_dataset,
        batch_size=64,
        shuffle=False)

    result = calc_accuracy_multi_advisor(model=model, loader=test_loader, args=args, verbose=True, hter=True)
    print(result)


if __name__ == '__main__':
    pretrain_dir = "../output/models/resnet18_se_no_dropout_no_seed_deeppix_7_rgb.pth"
    args.modal = 'rgb'
    args.batch_size=32
    args.gpu = 2
    args.data_root = ""
    args.backbone = "resnet18_se"
    model = SURF_Deeppix(args)
    model.load_state_dict(torch.load(pretrain_dir))

    batch_test(model=model, args=args)
