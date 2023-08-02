from src.surf_baseline_multi_dataloader import surf_multi_transforms_train
from src.surf_baseline_single_dataloader import surf_single_transforms_train, surf_single_transforms_test
from datasets.surf_single_txt import SURF_Single
from datasets.surf_txt import SURF
from datasets.surf_admd_kd import SURF_ADMD_KD

import torch


def surf_admd_kd_dataloader(state, args):
    '''

    :param train:
    :param args:
    :return:
    '''

    # dataset for student
    if state == 'train':
        txt_dir = args.data_root + '/train_list.txt'
        # txt_dir = args.data_root + '/Training.txt'
        root_dir = args.data_root
        admd_dataset = SURF_ADMD_KD(txt_dir=txt_dir,
                                    root_dir=root_dir,
                                    transform=surf_single_transforms_train)


    elif state == 'test':
        txt_dir = args.data_root + '/test_private_list.txt'
        # txt_dir = args.data_root + '/Training.txt'
        root_dir = args.data_root
        admd_dataset = SURF_ADMD_KD(txt_dir=txt_dir,
                                    root_dir=root_dir,
                                    transform=surf_single_transforms_test)
    else:
        print("surf_admd_kd_dataloader")
        admd_dataset = None

    admd_kd_data_loader = torch.utils.data.DataLoader(
        dataset=admd_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )

    return admd_kd_data_loader
