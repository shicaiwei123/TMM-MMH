from datasets.surf_sift_kd import Resize_multi, Normaliztion_multi, ToTensor_multi, RandomHorizontalFlip_multi, \
    RondomCrop_multi, RondomRotion_multi, Cutout_multi
from datasets.surf_sift_kd import SURF_SIFT

import torchvision.transforms as tt
import torch

surf_sift_patch_transforms_train = tt.Compose(
    [
        Resize_multi((144, 144)),
        RondomRotion_multi(30),
        RondomCrop_multi(112),
        RandomHorizontalFlip_multi(),
        ToTensor_multi(),
        # Cutout_multi(30),
        Normaliztion_multi(),

    ]
)

surf_sift_patch_transforms_test = tt.Compose(
    [
        # Resize_multi((144, 144)),
        # RondomCrop_multi(112),
        Resize_multi((112, 112)),
        # RandomHorizontalFlip_multi(),
        ToTensor_multi(),
        Normaliztion_multi(),
    ]
)


def surf_sift_patch_dataloader(train, args):
    # dataset and data loader
    if train:
        # txt_dir = args.data_root + '/train_list.txt'
        # root_dir = args.data_root
        txt_dir="/home/shicaiwei/data/liveness_data/CASIA-SUFR/Training.txt"
        root_dir="/home/shicaiwei/data/liveness_data/CASIA-SUFR"
        surf_dataset = SURF_SIFT(txt_dir=txt_dir,
                                 root_dir=root_dir,
                                 args=args,
                                 transform=surf_sift_patch_transforms_train)

    else:
        # txt_dir = args.data_root + '/val_private_list.txt'
        # root_dir = args.data_root
        txt_dir="/home/shicaiwei/data/liveness_data/CASIA-SUFR/Training.txt"
        root_dir="/home/shicaiwei/data/liveness_data/CASIA-SUFR"

        surf_dataset = SURF_SIFT(txt_dir=txt_dir,
                                 root_dir=root_dir,
                                 args=args,
                                 transform=surf_sift_patch_transforms_train)

    surf_data_loader = torch.utils.data.DataLoader(
        dataset=surf_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )

    return surf_data_loader
