import torchvision.transforms as tt
import torch

from datasets.cefa_single_protocol import CEFA_Single
from lib.processing_utils import get_mean_std
import datasets.cefa_dataset_class as data

cefa_single_transforms_train = tt.Compose(
    [
        # tt.RandomRotation(30),
        tt.Resize((144, 144)),
        tt.RandomHorizontalFlip(),
        # tt.ColorJitter(brightness=0.3),
        tt.RandomCrop((112, 112)),
        tt.ToTensor(),
        # tt.RandomErasing(p=0.5, scale=(0.05, 0.33)),
        # tt.Normalize(mean=[0.5, 0.5, 0.5, ], std=[0.5, 0.5, 0.5, ])
    ]
)

cefa_single_transforms_train_cycle = tt.Compose(
    [
        tt.RandomRotation(10),
        tt.Resize((128, 128)),
        tt.RandomHorizontalFlip(),
        tt.ToTensor(),
        # tt.Normalize(mean=[0.5, 0.5, 0.5, ], std=[0.5, 0.5, 0.5, ])
    ]
)

# from PIL import Image
#
# img_pil = Image.open(
#     "/home/shicaiwei/data/liveness_data/CASIA-SUFR/Training/real_part/CLKJ_AS0137/real.rssdk/depth/31.jpg").convert(
#     'RGB')
# img_r = cefa_single_transforms_train(img_pil)
# img_b = cefa_single_transforms_train(img_pil)
# img_r.show()
# img_b.show()


cefa_single_transforms_test = tt.Compose(
    [
        # tt.Resize((144, 144)),
        # tt.RandomCrop((112, 112)),
        tt.Resize((112, 112)),
        tt.ToTensor(),
        # tt.Normalize(mean=[0.5, 0.5, 0.5, ], std=[0.5, 0.5, 0.5, ])
    ]
)

cefa_single_transforms_test_cycle = tt.Compose(
    [
        # tt.Resize((144, 144)),
        # tt.RandomCrop((112, 112)),
        tt.Resize((128, 128)),
        tt.ToTensor(),
        # tt.Normalize(mean=[0.5, 0.5, 0.5, ], std=[0.5, 0.5, 0.5, ])
    ]
)


def cefa_baseline_single_dataloader(train, args):
    # dataset and data loader

    if train:

        cefa_dataset = CEFA_Single(args=args, modal=args.modal, mode='train', protocol=args.protocol,
                                   transform=cefa_single_transforms_train)
        cefa_data_loader = torch.utils.data.DataLoader(
            dataset=cefa_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4
        )

    else:
        cefa_dataset = CEFA_Single(args=args, modal=args.modal, mode='dev', protocol=args.protocol,
                                   transform=cefa_single_transforms_test)
        cefa_data_loader = torch.utils.data.DataLoader(
            dataset=cefa_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4
        )

    return cefa_data_loader
