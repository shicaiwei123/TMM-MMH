import sys

sys.path.append('..')
import torch
from itertools import chain
import os

from datasets.surf_sift_kd import SURF_SIFT
from src.surf_sift_patch_kd_dataloader import surf_sift_patch_dataloader,surf_sift_patch_transforms_test
from models.surf_sift_patch import SURF_SIFT_Patch
from models.resnet_se_sift_patch import resnet18_se_sift_patch
from loss.kd import *
from lib.model_develop import train_knowledge_distill_sift_patch
from configuration.config_sift_patch_kd import args


def deeppix_main(args):
    args.modal = args.student_modal  # 用于获取指定的模态训练学生模型

    train_loader = surf_sift_patch_dataloader(train=True, args=args)

    # 利用multi-modal 训练过程中rgb图像
    test_loader = surf_sift_patch_dataloader(train=False, args=args)


    # seed_torch(2)
    args.log_name = args.name + '.csv'
    args.model_name = args.name

    args.modal = args.teacher_modal
    teacher_model = SURF_SIFT_Patch(args)
    args.modal = args.student_modal
    student_model = resnet18_se_sift_patch(args)

    # 初始化并且固定teacher 网络参数
    teacher_model.load_state_dict(
        torch.load(os.path.join(args.model_root, 'resnet18_dropout_no_seed_no_share_multi.pth')))
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

    train_knowledge_distill_sift_patch(net_dict=nets, cost_dict=criterions, optimizer=optimizer, train_loader=train_loader,
                                  test_loader=test_loader,
                                  args=args)


if __name__ == '__main__':
    deeppix_main(args)
