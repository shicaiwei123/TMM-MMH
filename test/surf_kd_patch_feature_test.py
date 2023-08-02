import sys

sys.path.append('..')
from models.resnet_se_patch_feature import resnet18_se_patch_feature
from src.surf_baseline_multi_dataloader import surf_multi_transforms_train, surf_multi_transforms_test
from lib.model_develop import calc_accuracy_kd_patch_feature
from datasets.surf_txt import SURF
from configuration.config_patch_kd import args
import torch
import torch.nn as nn
from lib.processing_utils import save_csv


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

    result, output_logits = calc_accuracy_kd_patch_feature(model=model, loader=test_loader, args=args, verbose=True,
                                                           hter=True)

    output_logits=output_logits.cpu().detach().numpy()
    print(output_logits.shape)
    save_csv("surf_dad_prediction.txt", output_logits)

    print(result)


if __name__ == '__main__':
    pretrain_dir = "../output/models/surf_patch_feature_l3_l4_multi_multi_rgb_lr_0.001_version_1_lambda_100.0_hter_best_.pth"
    args.gpu = 0
    args.miss_modal = 0
    args.backbone = "resnet18_se"
    args.modal = 'rgb'
    args.student_data = 'multi_rgb'
    args.sift = False
    model = resnet18_se_patch_feature(args)
    model.load_state_dict(torch.load(pretrain_dir))

    batch_test(model=model, args=args)
