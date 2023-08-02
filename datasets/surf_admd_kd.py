from datasets.surf_txt import SURF

from torch.utils.data import Dataset, DataLoader
from datasets.surf_single_txt import SURF_Single


# 将三个数据集绑定在一起,用于admd方法的测试,但是适当调整训练的代码,也可以适用于训练代码
class SURF_ADMD_KD(Dataset):
    '''
    ADMD method 测试数据集
    '''

    def __init__(self, txt_dir, root_dir, transform=None):
        self.rgb = SURF_Single(txt_dir, root_dir, transform=transform, modal='rgb')
        self.ir = SURF_Single(txt_dir, root_dir, transform=transform, modal='ir')
        self.depth = SURF_Single(txt_dir, root_dir, transform=transform, modal='depth')

    def __len__(self):
        return len(self.rgb)

    def __getitem__(self, idx):
        rgb, label = self.rgb[idx]
        ir, label = self.ir[idx]
        depth, label = self.depth[idx]

        return rgb, ir, depth, label
