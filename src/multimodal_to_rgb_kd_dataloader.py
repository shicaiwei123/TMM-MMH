from src.surf_baseline_multi_dataloader import surf_multi_transforms_train
from src.surf_baseline_single_dataloader import surf_single_transforms_train, surf_single_transforms_test
from datasets.surf_single_txt import SURF_Single
from datasets.surf_txt import SURF
from datasets.surf_kd import SURF_KD

import torch


def multimodal_to_rgb_kd_dataloader(train, args):
    '''

    :param train:
    :param args:
    :return:
    '''
    # dataset for multimodal(teacher)
    txt_dir = args.data_root + '/train_list.txt'
    # txt_dir = args.data_root + '/Training.txt'
    root_dir = args.data_root
    teacher_dataset = SURF(txt_dir=txt_dir,
                           root_dir=root_dir,
                           transform=surf_multi_transforms_train, miss_modal=args.miss_modal)

    # dataset for student
    if train:
        txt_dir = args.data_root + '/train_list.txt'
        # txt_dir = args.data_root + '/Training.txt'
        root_dir = args.data_root
        student_dataset = SURF_Single(txt_dir=txt_dir,
                                      root_dir=root_dir,
                                      transform=surf_single_transforms_train, modal=args.modal)

        kd_dataset = SURF_KD(teacher_dataset, student_dataset, train)


    else:
        # txt_dir = args.data_root + '/val_private_list.txt'
        txt_dir = args.data_root + '/Training.txt'
        root_dir = args.data_root
        student_dataset = SURF_Single(txt_dir=txt_dir,
                                      root_dir=root_dir,
                                      transform=surf_single_transforms_test, modal=args.modal)

        kd_dataset = SURF_KD(teacher_dataset, student_dataset, train)

    kd_data_loader = torch.utils.data.DataLoader(
        dataset=kd_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )

    return kd_data_loader



