'''
code for https://arxiv.org/abs/1810.08437
the dataset used in train and test is different.specifically, when train, we don't care the data is paired or not, so we use the surf_baseline_single_dataloader for
to get different modal data. however, in the test time, the input should be paired, so, we need to rewrite a dataloader.

the surf_admd_dataloader.py is not used.

different with other kd methods, the train stage of this method contains two stage, need to rewrite a train function.

'''


import sys

sys.path.append('..')
import torch
import os

from src.surf_baseline_single_dataloader import surf_baseline_single_dataloader
from models.resnet_se_admd import resnet18_se_admd
from lib.model_develop import train_tgt
from lib.model_arch_utils import Discriminator
from configuration.config_admd_kd import args


def deeppix_main(args):
    args.modal = args.teacher_modal
    teacher_dataloader = surf_baseline_single_dataloader(train=True, args=args)
    args.modal = args.student_modal
    student_dataloader = surf_baseline_single_dataloader(train=True, args=args)

    # seed_torch(2)
    args.log_name = args.name + '.csv'
    args.model_name = args.name

    args.modal = args.teacher_modal
    teacher_model = resnet18_se_admd(args)
    args.modal = args.student_modal
    student_model = resnet18_se_admd(args)

    critical = Discriminator(args=args, input_dims=args.feature_dim)

    # 初始化并且固定teacher 网络参数
    if args.teacher_data == 'rgb':
        teacher_model.load_state_dict(
            torch.load(
                os.path.join(args.model_root, 'resnet18_se_dropout_no_seed_rgb.pth')))
    elif args.teacher_data == 'depth':
        teacher_model.load_state_dict(
            torch.load(
                os.path.join(args.model_root, 'resnet18_se_dropout_no_seed_depth.pth')))
    elif args.teacher_data == 'ir':
        teacher_model.load_state_dict(
            torch.load(
                os.path.join(args.model_root, 'resnet18_se_dropout_no_seed_ir.pth')))
    else:
        print("teacher_model init failure")

    # 固定参数不更新
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # 初始化学生网络
    if args.init_mode == 'rgb':
        print('--------------------------------init student_model with rgb modal--------------------------------------')
        student_model.load_state_dict(torch.load(os.path.join(args.model_root, 'resnet18_se_dropout_no_seed_rgb.pth')))
    elif args.init_mode == 'depth':
        print(
            '--------------------------------init student_model with depth modal--------------------------------------')
        student_model.load_state_dict(
            torch.load(os.path.join(args.model_root, 'resnet18_se_dropout_no_seed_depth.pth')))
    elif args.init_mode == 'ir':
        print(
            '--------------------------------init student_model with ir modal--------------------------------------')
        student_model.load_state_dict(
            torch.load(os.path.join(args.model_root, 'resnet18_se_dropout_no_seed_ir.pth')))

    # 如果有GPU
    if torch.cuda.is_available():
        teacher_model.cuda()  # 将所有的模型参数移动到GPU上
        student_model.cuda()
        critical = critical.cuda()
        print("GPU is using")

    train_tgt(src_encoder=teacher_model, tgt_encoder=student_model,
              critic=critical, src_data_loader=teacher_dataloader, tgt_data_loader=student_dataloader,
              args=args)


if __name__ == '__main__':
    deeppix_main(args)
