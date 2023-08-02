import torchvision.transforms as tt
import torch

from datasets.surf_multi_advisor import SURF_Multi_Advisor
from lib.processing_utils import get_mean_std

surf_deeppix_transforms_train = tt.Compose(
    [
        tt.RandomRotation(30),
        tt.Resize((144, 144)),
        tt.RandomHorizontalFlip(),
        # tt.ColorJitter(brightness=0.3),
        tt.RandomCrop((112, 112)),
        tt.ToTensor(),
        # tt.RandomErasing(p=0.5, scale=(0.05, 0.33)),
        tt.Normalize(mean=[0.5, 0.5, 0.5, ], std=[0.5, 0.5, 0.5, ])
    ]
)

# from PIL import Image
#
# img_pil = Image.open(
#     "/home/shicaiwei/data/liveness_data/CASIA-SUFR/Training/real_part/CLKJ_AS0137/real.rssdk/depth/31.jpg").convert(
#     'RGB')
# img_r = surf_deeppix_transforms_train(img_pil)
# img_b = surf_deeppix_transforms_train(img_pil)
# img_r.show()
# img_b.show()


surf_deeppix_transforms_test = tt.Compose(
    [
        tt.Resize((112, 112)),
        tt.ToTensor(),
        tt.Normalize(mean=[0.5, 0.5, 0.5, ], std=[0.5, 0.5, 0.5, ])
    ]
)


def surf_deeppix_dataloader(train, args):
    # dataset and data loader
    if train:
        txt_dir = args.data_root + '/train_list.txt'
        root_dir = args.data_root
        # txt_dir="/home/shicaiwei/data/liveness_data/CASIA-SUFR/Training.txt"
        # root_dir="/home/shicaiwei/data/liveness_data/CASIA-SUFR"
        surf_dataset = SURF_Multi_Advisor(txt_dir=txt_dir,
                                          root_dir=root_dir,
                                          args=args, transform=surf_deeppix_transforms_train)



    else:
        txt_dir = args.data_root + '/val_private_list.txt'
        root_dir = args.data_root
        # txt_dir="/home/shicaiwei/data/liveness_data/CASIA-SUFR/Training.txt"
        # root_dir="/home/shicaiwei/data/liveness_data/CASIA-SUFR"
        surf_dataset = SURF_Multi_Advisor(txt_dir=txt_dir,
                                          root_dir=root_dir,
                                          args=args, transform=surf_deeppix_transforms_test)

    surf_data_loader = torch.utils.data.DataLoader(
        dataset=surf_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )

    return surf_data_loader
