import sys

sys.path.append('..')
from models.surf_baseline import SURF_Baseline
from src.surf_baseline_multi_dataloader import surf_multi_transforms_train, surf_multi_transforms_test
from lib.model_develop import calc_accuracy_kd_patch_feature
from datasets.surf_txt import SURF
from configuration.config_patch_kd import args
import torch
import torch.nn as nn
from lib.model_develop_utils import sne_analysis



def batch_test(model, args):
    '''
    利用dataloader 装载测试数据,批次进行测试
    :return:
    '''

    root_dir = "/home/shicaiwei/data/liveness_data/CASIA-SUFR"
    txt_dir = "/home/shicaiwei/data/liveness_data/CASIA-SUFR/val_private_list.txt"
    surf_dataset = SURF(txt_dir=txt_dir,
                        root_dir=root_dir,
                        transform=surf_multi_transforms_test, miss_modal=args.miss_modal)
    #
    # surf_dataset = SURF_generate(rgb_dir=args.rgb_root, depth_dir=args.depth_root, ir_dir=args.ir_root,
    #                              transform=surf_multi_transforms_test)

    test_loader = torch.utils.data.DataLoader(
        dataset=surf_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4
    )

    sne_analysis(model,test_loader)


if __name__ == '__main__':
    pretrain_dir = "/home/shicaiwei/project/multimodal_inheritance/multi_model_fas/output/models/resnet18_dropout_no_seed_3_multi.pth"
    args.gpu = 3
    args.miss_modal = 0
    args.backbone = "resnet18_se"
    args.modal = 'multi'
    args.student_data = 'multi_rgb'
    args.sift=False
    model = SURF_Baseline(args)
    # model.load_state_dict(torch.load(pretrain_dir))
    # model_child=[*model.children()]
    # model_select=[]
    # model_select.append(model_child[0])
    # model_select.append(model_child[1])
    # model_select.append(model_child[2])
    # model_select.append(model_child[3][0])
    # model_select.append(model_child[3][1])
    # model_select.append(model_child[3][2])
    # model=nn.Sequential(*model_select)

    batch_test(model=model, args=args)
