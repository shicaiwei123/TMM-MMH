''' 将casia-surf 数据集的三个模态分开变成三个数据集:surf_rgb,surf_depth,surf_ir,每个数据集都包含着真人和欺骗样本,并且被分为训练测试'''
'''
活体检测多模态数据caisa-surf 的dataloader
'''

# from skimage import io, transform
import cv2
from PIL import Image
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pdb
import math
import os
from lib.processing_utils import read_txt


class SURF_Multi_Advisor(Dataset):

    def __init__(self, txt_dir, root_dir, args, transform=None):
        self.related_sample_path_list = read_txt(txt_dir)
        self.root_dir = root_dir
        self.transform = transform
        self.args = args

    def __len__(self):
        return len(self.related_sample_path_list)

    def __getitem__(self, idx):
        related_sample_path = self.related_sample_path_list[idx]
        related_sample_path_split = related_sample_path.split(" ")

        rgb_path = os.path.join(self.root_dir, related_sample_path_split[0])
        depth_path = os.path.join(self.root_dir, related_sample_path_split[1])
        ir_path = os.path.join(self.root_dir, related_sample_path_split[2])

        binary_label = np.int(related_sample_path_split[3])
        # print(binary_label)

        image_rgb = Image.open(rgb_path).convert('RGB')
        image_ir_gray = cv2.imread(ir_path, 0)
        image_ir_gray = cv2.resize(image_ir_gray, (32, 32))
        image_ir_gray = np.float32(image_ir_gray) / 255
        image_depth_gray = cv2.imread(depth_path, 0)
        image_depth_gray = cv2.resize(image_depth_gray, (32, 32))
        image_depth_gray = np.float32(image_depth_gray) / 255

        if self.args.origin_deeppix:
            if binary_label == 0:
                image_ir_gray = np.zeros((32, 32))
                image_depth_gray = np.zeros((32, 32))
            else:
                image_ir_gray = np.ones((32, 32))
                image_depth_gray = np.ones((32, 32))

        if self.transform:
            image_rgb = self.transform(image_rgb)

        sample = {'image_rgb': image_rgb, 'image_ir': image_depth_gray, 'image_depth': image_ir_gray,
                  'binary_label': binary_label}
        return sample
