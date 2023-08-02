'''
活体检测多模态数据caisa-surf 的dataloader
'''

# from skimage import io, transform
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pdb
import math
import os
from lib.processing_utils import read_txt, get_file_list
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as tt
from sklearn.cluster import KMeans
from time import time
import warnings


class RandomHorizontalFlip_multi(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image_x, image_ir, image_depth, binary_label, cluster_center = sample['image_x'], sample['image_ir'], sample[
            'image_depth'], sample['binary_label'], sample['cluster_center']

        if random.random() < self.p:
            new_image_x = cv2.flip(image_x, 1)
            new_image_ir = cv2.flip(image_ir, 1)
            new_image_depth = cv2.flip(image_depth, 1)
            return {'image_x': new_image_x, 'image_ir': new_image_ir, 'image_depth': new_image_depth,
                    'binary_label': binary_label, 'cluster_center': cluster_center}
        else:
            return sample


class Resize_multi(object):

    def __init__(self, size):
        '''
        元组size,如(112,112)
        :param size:
        '''
        self.size = size

    def __call__(self, sample):
        image_x, image_ir, image_depth, binary_label, cluster_center = sample['image_x'], sample['image_ir'], sample[
            'image_depth'], sample['binary_label'], sample['cluster_center']

        new_image_x = cv2.resize(image_x, self.size)
        new_image_ir = cv2.resize(image_ir, self.size)
        new_image_depth = cv2.resize(image_depth, self.size)

        return {'image_x': new_image_x, 'image_ir': new_image_ir, 'image_depth': new_image_depth,
                'binary_label': binary_label, 'cluster_center': cluster_center}


class RondomRotion_multi(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, sample):
        image_x, image_ir, image_depth, binary_label, cluster_center = sample['image_x'], sample['image_ir'], sample[
            'image_depth'], sample['binary_label'], sample['cluster_center']

        (h, w) = image_x.shape[:2]
        (cx, cy) = (w / 2, h / 2)

        # 设置旋转矩阵
        angle = random.randint(-self.angle, self.angle)
        M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
        cos = np.abs(M[0, 0]) * 0.8
        sin = np.abs(M[0, 1]) * 0.8

        # 计算图像旋转后的新边界
        nw = int((h * sin) + (w * cos))
        nh = int((h * cos) + (w * sin))

        # 调整旋转矩阵的移动距离（t_{x}, t_{y}）
        M[0, 2] += (nw / 2) - cx
        M[1, 2] += (nh / 2) - cy

        new_image_x = cv2.warpAffine(image_x, M, (nw, nh))
        new_image_ir = cv2.warpAffine(image_ir, M, (nw, nh))
        new_image_depth = cv2.warpAffine(image_depth, M, (nw, nh))

        return {'image_x': new_image_x, 'image_ir': new_image_ir, 'image_depth': new_image_depth,
                'binary_label': binary_label, 'cluster_center': cluster_center}


class RondomCrop_multi(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        image_x, image_ir, image_depth, binary_label, cluster_center = sample['image_x'], sample['image_ir'], sample[
            'image_depth'], sample['binary_label'], sample['cluster_center']

        h, w = image_x.shape[:2]

        y = np.random.randint(0, h - self.size)
        x = np.random.randint(0, w - self.size)

        new_image_x = image_x[y:y + self.size, x:x + self.size, :]
        new_image_ir = image_ir[y:y + self.size, x:x + self.size, :]
        new_image_depth = image_depth[y:y + self.size, x:x + self.size, :]

        return {'image_x': new_image_x, 'image_ir': new_image_ir, 'image_depth': new_image_depth,
                'binary_label': binary_label, 'cluster_center': cluster_center}


class Cutout_multi(object):
    '''
    作用在to tensor 之后
    '''

    def __init__(self, length=30):
        self.length = length

    def __call__(self, sample):
        image_x, image_ir, image_depth, binary_label, cluster_center = sample['image_x'], sample['image_ir'], sample[
            'image_depth'], sample['binary_label'], sample['cluster_center']
        h, w = image_x.shape[1], image_x.shape[2]  # Tensor [1][2],  nparray [0][1]
        length_new = np.random.randint(1, self.length)
        y = np.random.randint(h - length_new)
        x = np.random.randint(w - length_new)

        image_x[y:y + length_new, x:x + length_new] = 0
        image_ir[y:y + length_new, x:x + length_new] = 0
        image_depth[y:y + length_new, x:x + length_new] = 0

        return {'image_x': image_x, 'image_ir': image_ir, 'image_depth': image_depth, 'binary_label': binary_label,
                'cluster_center': cluster_center}


class Normaliztion_multi(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """

    def __init__(self):
        self.a = 1

    def __call__(self, sample):
        image_x, image_ir, image_depth, binary_label, cluster_center = sample['image_x'], sample['image_ir'], sample[
            'image_depth'], sample['binary_label'], sample['cluster_center']

        new_image_x = (image_x - 127.5) / 128  # [-1,1]
        new_image_ir = (image_ir - 127.5) / 128  # [-1,1]
        new_image_depth = (image_depth - 127.5) / 128  # [-1,1]

        return {'image_x': new_image_x, 'image_ir': new_image_ir, 'image_depth': new_image_depth,
                'binary_label': binary_label, 'cluster_center': cluster_center}


class ToTensor_multi(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __init__(self):
        self.a = 1

    def __call__(self, sample):
        image_x, image_ir, image_depth, binary_label, cluster_center = sample['image_x'], sample['image_ir'], sample[
            'image_depth'], sample['binary_label'], sample['cluster_center']

        # swap color axis because    BGR2RGB
        # numpy image: (batch_size) x T x H x W x C
        # torch image: (batch_size) x T x C X H X W
        image_x = image_x.transpose((2, 0, 1))
        image_x = np.array(image_x)

        image_ir = image_ir.transpose((2, 0, 1))
        image_ir = np.array(image_ir)

        image_depth = image_depth.transpose((2, 0, 1))
        image_depth = np.array(image_depth)

        binary_label = np.array(binary_label)
        cluster_center = np.array(cluster_center)
        try:
            return {'image_x': torch.from_numpy(image_x.astype(np.float)).float(),
                    'image_ir': torch.from_numpy(image_ir.astype(np.float)).float(),
                    'image_depth': torch.from_numpy(image_depth.astype(np.float)).float(),
                    'binary_label': torch.from_numpy(binary_label),
                    'cluster_center': torch.from_numpy(cluster_center)}
        except Exception as e:
            print(e)


class SURF_SIFT(Dataset):

    def __init__(self, txt_dir, root_dir, args, transform=None):
        self.related_sample_path_list = read_txt(txt_dir)

        # img_index = []
        #
        # for idx in range(len(self.related_sample_path_list)):
        #     related_sample_path = self.related_sample_path_list[idx]
        #     related_sample_path_split = related_sample_path.split(" ")
        #     rgb_related_sample_path_split = related_sample_path_split[0]
        #     img_index.append(rgb_related_sample_path_split.split('/')[-1])
        #
        # # 利用img_index对self.related_sample_path_list排序,使得能够按照顺序读取txt
        # img_index, self.related_sample_path_list = (list(t) for t in
        #                                             zip(*sorted(zip(img_index, self.related_sample_path_list))))

        self.root_dir = root_dir
        self.transform = transform
        self.kmeans_modal = KMeans(n_clusters=args.cluster_num, random_state=9)
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.args = args
        self.image_cache = None
        self.cluster_center_cache = np.ones((16, 2)) * 2

    def __len__(self):
        return len(self.related_sample_path_list)
        # return 5

    def __getitem__(self, idx):
        related_sample_path = self.related_sample_path_list[idx]
        # print(related_sample_path)
        related_sample_path_split = related_sample_path.split(" ")

        rgb_path = os.path.join(self.root_dir, related_sample_path_split[0])
        depth_path = os.path.join(self.root_dir, related_sample_path_split[1])
        ir_path = os.path.join(self.root_dir, related_sample_path_split[2])

        binary_label = np.int(related_sample_path_split[3])

        # print(rgb_path)
        # print(ir_path)
        # print(depth_path)
        image_rgb = cv2.imread(rgb_path)
        image_ir = cv2.imread(ir_path)
        image_depth = cv2.imread(depth_path)

        image_rgb, cluster_center = self.singe_image_analysis(image_rgb)

        sample = {'image_x': image_rgb, 'image_ir': image_depth, 'image_depth': image_ir, 'binary_label': binary_label,
                  'cluster_center': cluster_center}

        if self.transform:
            sample = self.transform(sample)

        # print(sample)
        return sample

    def singe_image_analysis(self, img):

        kpt_list = []
        cluster_center = []
        begin_time = time()

        im = cv2.resize(img, (112, 112))

        # 检测关键点
        kpts = self.sift.detect(im)

        # 关键地聚类
        for data in kpts:
            position = data.pt
            position = np.array(position)
            kpt_list.append(position)

        if len(kpt_list) < self.args.cluster_num:
            # 绘图
            # kp_image = cv2.drawKeypoints(im, kpts, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # cv2.imshow('image', kp_image)
            # cv2.waitKey(0)

            cluster_center = self.cluster_center_cache
            # img = self.image_cache
        else:

            kpt_cluster = self.kmeans_modal.fit_predict(kpt_list)

            # 求聚类中心的坐标
            for i in range(self.args.cluster_num):
                kpt_arr = np.array(kpt_list)
                kpt_cluster = np.array(kpt_cluster)
                data = kpt_arr[kpt_cluster == i, :]
                with warnings.catch_warnings(record=True) as w:
                    data_mean = data.mean(axis=0)
                    # 一旦检测到警告,终止聚类,使用缓存当做聚类结果
                    if len(w) > 0:
                        cluster_center = self.cluster_center_cache
                        break
                    else:
                        cluster_center.append(data_mean)
            cluster_center = np.array(cluster_center)
            self.cluster_center_cache = cluster_center

            end_time = time()
            # print(begin_time - end_time)

        return img, cluster_center
