import sys

sys.path.append('..')
from models.sp_ap_ensemble import SAP
from src.surf_baseline_multi_dataloader import surf_multi_transforms_train, surf_multi_transforms_test
from lib.model_develop import calc_accuracy_ensemble
from datasets.surf_txt import SURF, SURF_generate
from configuration.config_sp_ap_ensemble import args
import torch
import torch.nn as nn
import os


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
        shuffle=False)

    result = calc_accuracy_ensemble(model=model, loader=test_loader, verbose=True, hter=True)
    print(result)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(3)

    #sp_ensemble_ap_multiKD_acer_best__lr_0.0001_version_2_acer_best_.pth

    for i in range(50):
        try:
            i = i + 0
            pretrain_dir = "../output/models/sp_ensemble_ap_multiKD_acer_best__lr_0.0001_version_" + str(9) + "_" + str(
                i) + "_.pth"
            args.gpu = 3
            args.backbone = "resnet18_se"
            model = SAP(args)
            test_para = torch.load(pretrain_dir)
            model.load_state_dict(torch.load(pretrain_dir))

            batch_test(model=model, args=args)
        except Exception as e:
            print(1)
            continue
