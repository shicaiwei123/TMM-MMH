from datasets.surf_txt import SURF
from datasets.surf_single_txt import SURF_Single

from torch.utils.data import Dataset, DataLoader


# 将两个数据集绑定在一起
class SURF_KD(Dataset):
    def __init__(self, dataset_multi, dataset_single, train):
        self.dataset_multi = dataset_multi
        self.dataset_single = dataset_single
        self.train = train

    def __len__(self):
        if self.train:
            return len(self.dataset_multi)
        else:
            return len(self.dataset_single)

    def __getitem__(self, idx):
        if self.train:
            sample = self.dataset_multi[idx]
            single_data, label = self.dataset_single[idx]
            return sample, single_data, label

        # 测试过程只加载测试数据.
        else:
            sample = self.dataset_multi[idx]
            single_data, label = self.dataset_single[idx]
            return sample, single_data, label