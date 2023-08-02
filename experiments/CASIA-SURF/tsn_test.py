
import sys

sys.path.append('..')
from models.resnet_se_patch_feature import resnet18_se_patch_feature
from src.surf_baseline_multi_dataloader import surf_multi_transforms_train, surf_multi_transforms_test
from src.surf_baseline_single_dataloader import surf_single_transforms_test
from lib.model_develop import calc_accuracy_kd_patch_feature
from datasets.surf_txt import SURF
from datasets.surf_single_txt import SURF_Single
from configuration.config_patch_kd import args
import torch
import torch.nn as nn
from lib.model_develop_utils import sne_analysis

def batch_test_single(args):
    '''
    利用dataloader 装载测试数据,批次进行测试
    :return:
    '''

    root_dir = "/home/shicaiwei/data/liveness_data/CASIA-SUFR"
    txt_dir = "/home/shicaiwei/data/liveness_data/CASIA-SUFR/val_private_list.txt"
    surf_dataset = SURF_Single(txt_dir=txt_dir,
                        root_dir=root_dir,
                        transform=surf_single_transforms_test)
    #
    # surf_dataset = SURF_generate(rgb_dir=args.rgb_root, depth_dir=args.depth_root, ir_dir=args.ir_root,
    #                              transform=surf_multi_transforms_test)

    test_loader = torch.utils.data.DataLoader(
        dataset=surf_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4
    )


    pretrain_dir = "/home/shicaiwei/project/multimodal_inheritance/multi_model_fas/output/models/resnet18_se_dropout_no_seed_rgb_version_5.pth"
    args.gpu = 3
    args.miss_modal = 0
    args.backbone = "resnet18_se"
    args.modal = 'rgb'
    args.student_data = 'multi_rgb'
    args.sift=False

    model = resnet18_se_patch_feature(args)
    model.load_state_dict(torch.load(pretrain_dir))
    model_child=[*model.children()][0:-2]
    model=nn.Sequential(*model_child)



    sne_analysis(model=model,data_loader=test_loader)




if __name__ == '__main__':


    batch_test_single(args)
