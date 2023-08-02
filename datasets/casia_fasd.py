import cv2
import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from lib.processing_utils import get_file_list
from lib.processing_utils import FaceDection
from PIL import Image


class CASIA_Dataset(Dataset):

    def __init__(self, living_dir, spoofing_dir, args, balance, data_transform=None, sampe_interal=1):

        self.living_path_list = get_file_list(living_dir)
        self.spoofing_path_list = get_file_list(spoofing_dir)

        # 间隔取样,控制数量
        if balance:
            self.spoofing_path_list = sorted(self.spoofing_path_list)
            balance_factor = int(np.round(len(self.spoofing_path_list) / len(self.living_path_list)))
            if balance_factor < 1:
                balance_factor = 1
            self.spoofing_path_list = self.spoofing_path_list[0:len(self.spoofing_path_list):balance_factor]

        self.spoofing_path_list = self.spoofing_path_list[0:len(self.spoofing_path_list):sampe_interal]
        self.living_path_list = self.living_path_list[0:len(self.living_path_list):sampe_interal]

        self.img_path_list = self.spoofing_path_list + self.living_path_list
        self.data_transform = data_transform
        # self.face_detector = FaceDection("")
        self.args = args

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):

        img_path = self.img_path_list[idx]
        img_path_split = img_path.split('/')

        # label
        if img_path_split[-2] == 'spoofing' or img_path_split[-3] == 'spoofing':

            spoofing_label = 0

        else:
            spoofing_label = 1

        # read
        # print(img_path)
        image_rgb = cv2.imread(img_path)
        image_ir = cv2.imread(img_path)
        image_depth = cv2.imread(img_path)

        sample = {'image_x': image_rgb, 'image_ir': image_ir, 'image_depth': image_depth,
                  'binary_label': spoofing_label}

        if self.data_transform:
            sample = self.data_transform(sample)
        return sample


class CASIA_Single(Dataset):
    def __init__(self, living_dir, spoofing_dir, args, balance, data_transform=None, sampe_interal=1):

        self.living_path_list = get_file_list(living_dir)
        self.spoofing_path_list = get_file_list(spoofing_dir)

        # 间隔取样,控制数量
        if balance:
            self.spoofing_path_list = sorted(self.spoofing_path_list)
            balance_factor = int(np.round(len(self.spoofing_path_list) / len(self.living_path_list)))
            if balance_factor < 1:
                balance_factor = 1
            self.spoofing_path_list = self.spoofing_path_list[0:len(self.spoofing_path_list):balance_factor]

        self.spoofing_path_list = self.spoofing_path_list[0:len(self.spoofing_path_list):sampe_interal]
        self.living_path_list = self.living_path_list[0:len(self.living_path_list):sampe_interal]

        self.img_path_list = self.spoofing_path_list + self.living_path_list
        self.data_transform = data_transform
        # self.face_detector = FaceDection("")
        self.args = args

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):

        img_path = self.img_path_list[idx]
        img_path_split = img_path.split('/')

        # label
        if img_path_split[-2] == 'spoofing' or img_path_split[-3] == 'spoofing':

            spoofing_label = 0

        else:
            spoofing_label = 1

        image_rgb = Image.open(img_path).convert('RGB')

        if self.data_transform:
            image_rgb = self.data_transform(image_rgb)
        return image_rgb, spoofing_label
