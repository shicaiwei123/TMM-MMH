import sys

sys.path.append('..')
import torch
from itertools import chain
import os

from src.multimodal_to_rgb_kd_dataloader import multimodal_to_rgb_kd_dataloader
from src.surf_baseline_multi_dataloader import surf_baseline_multi_dataloader
from models.surf_patch import SURF_Patch
from models.resnet_se_patch_attention import resnet18_se_patch_attention
from models.resnet_se_patch import resnet18_se_patch
from loss.kd import *
from lib.model_develop import train_self_distill
from configuration.config_sp_to_patch import args
from lib.processing_utils import seed_torch


def deeppix_main(args):
    args.modal = args.student_modal  # 用于获取指定的模态训练学生模型

    train_loader = multimodal_to_rgb_kd_dataloader(train=True, args=args)

    if args.student_data == 'multi_rgb':  # 利用multi-modal 训练过程中rgb图像
        test_loader = surf_baseline_multi_dataloader(train=False, args=args)
    else:  # 利用单模态处理的rgb图像
        test_loader = multimodal_to_rgb_kd_dataloader(train=False, args=args)

    # seed_torch(2)
    args.log_name = args.name + '.csv'
    args.model_name = args.name

    args.modal = args.teacher_modal
    teacher_model = resnet18_se_patch_attention(args)
    args.modal = args.student_modal
    student_model = resnet18_se_patch_attention(args)

    # 初始化并且固定teacher 网络参数
    teacher_model.load_state_dict(
        torch.load(
            os.path.join(args.model_root, 'patch_feature_kd_mmd_avg_sp_multi_multi_rgb_lr_0.001_version_7_sift_0.pth')))
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False



    if args.init_mode == 'rgb':
        print('--------------------------------init student_model with rgb modal--------------------------------------')
        student_model.load_state_dict(torch.load(os.path.join(args.model_root, 'resnet18_se_dropout_no_seed_rgb.pth')))
    elif args.init_mode == 'depth':
        print(
            '--------------------------------init student_model with depth modal--------------------------------------')
        student_model.load_state_dict(
            torch.load(os.path.join(args.model_root, 'resnet18_se_dropout_no_seed_depth.pth')))
    elif args.init_mode == 'ap':
        print(
            '--------------------------------init student_model with sp feature--------------------------------------')
        para_dict = torch.load(os.path.join(args.model_root,
                                            'patch_kd_mmd_avg_multi_multi_rgb_lr_0.001_version_3_weight_patch_1.pth'))
        init_para_dict = student_model.state_dict()
        try:
            for k, v in para_dict.items():
                if k in init_para_dict:
                    init_para_dict[k] = para_dict[k]
            student_model.load_state_dict(init_para_dict)
        except Exception as e:
            print(e)

    # 如果有GPU
    if torch.cuda.is_available():
        teacher_model.cuda()  # 将所有的模型参数移动到GPU上
        student_model.cuda()
        print("GPU is using")

    # define loss functions
    if args.kd_mode == 'logits':
        criterionKD = Logits()
    elif args.kd_mode == 'st':
        criterionKD = SoftTarget(args.T)
    elif args.kd_mode == 'at':
        criterionKD = AT(args.p)
    elif args.kd_mode == 'fitnet':
        criterionKD = Hint()

    else:
        raise Exception('Invalid kd mode...')
    if args.cuda:
        criterionCls = torch.nn.CrossEntropyLoss().cuda()
    else:
        criterionCls = torch.nn.CrossEntropyLoss()

    # initialize optimizer

    if args.optim == 'sgd':
        print('--------------------------------optim with sgd--------------------------------------')
        if args.kd_mode in ['vid', 'ofd', 'afd']:
            optimizer = torch.optim.SGD(chain(student_model.parameters(),
                                              *[c.parameters() for c in criterionKD[1:]]),
                                        lr=args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay,
                                        nesterov=True)
        else:
            optimizer = torch.optim.SGD(student_model.parameters(),
                                        lr=args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay,
                                        nesterov=True)
    elif args.optim == 'adam':
        print('--------------------------------optim with adam--------------------------------------')
        if args.kd_mode in ['vid', 'ofd', 'afd']:
            optimizer = torch.optim.Adam(chain(student_model.parameters(),
                                               *[c.parameters() for c in criterionKD[1:]]),
                                         lr=args.lr,
                                         weight_decay=args.weight_decay,
                                         )
        else:
            optimizer = torch.optim.Adam(student_model.parameters(),
                                         lr=args.lr,
                                         weight_decay=args.weight_decay,
                                         )
    else:
        print('optim error')
        optimizer = None

    # warp nets and criterions for train and test
    nets = {'snet': student_model, 'tnet': teacher_model}
    criterions = {'criterionCls': criterionCls, 'criterionKD': criterionKD}

    train_self_distill(net_dict=nets, cost_dict=criterions, optimizer=optimizer, train_loader=train_loader,
                       test_loader=test_loader,
                       args=args)


if __name__ == '__main__':
    deeppix_main(args)
