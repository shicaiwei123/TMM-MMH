
import sys
from torch.utils.data import Dataset

from lib.processing_utils import read_txt, replace_string
import glob
from PIL import Image
import numpy as np
from action_utils import util
import datetime
import torch
import os
import pickle


class UCF101(Dataset):
    def __init__(self, data_root, split_path, clip_len, mode, sample_interal=2):
        '''
        动作的label直接从文件夹的名字中提取:a01_s01_e01   a01 的01
        :param data_root: 存放NW-HMDB51 数据集的路径
        :param split_path: path for split txt
        :param clip_len: 每个动作取多少帧用于训练
        :mode : rgb or depth
        :param train: 训练还是测试
        :param sample_interal: 数据抽样加快训练
        '''
        super(UCF101, self).__init__()

        self.test_data = []
        self.train_data = []

        self.clip_len = clip_len
        frames_path_len_min = 10000
        self.mode = mode

        txt_name = split_path.split('/')[-1]
        if 'train' in txt_name:
            self.train = True
        elif 'test' in txt_name:
            self.train = False
        else:
            print('error file')

        split_line = read_txt(split_path)
        for line in split_line:
            related_path = line.split(' ')[0]
            related_path = related_path.split('.')[0]
            label = int(line.split(' ')[1])-1
            video_path = os.path.join(data_root, related_path)

            img_list = os.listdir(video_path)
            img_len = len(img_list)
            # print(img_len,img_len/3)
            assert np.mod(img_len, 3) == 1
            single_modality_len = img_len // 3
            rgb_list = []
            flow_x_list = []
            flow_y_list = []
            for i in range(single_modality_len):
                i = str(i)
                i = i.zfill(5)
                rgb_list.append(os.path.join(video_path, "img_" + i + ".jpg"))
                flow_x_list.append(os.path.join(video_path, "flow_x_" + i + ".jpg"))
                flow_y_list.append(os.path.join(video_path, "flow_y_" + i + ".jpg"))
            if self.train:
                self.train_data.append(
                    {"frame": rgb_list, "flow_x": flow_x_list, "flow_y": flow_y_list, "label": label})
            else:
                self.test_data.append(
                    {"frame": rgb_list, "flow_x": flow_x_list, "flow_y": flow_y_list, "label": label})

        self.loader = lambda fl: Image.open(fl)

        if self.train:
            self.data = self.train_data
        else:
            self.data = self.test_data

        if self.train:
            self.clip_transform = util.clip_transform_ucf101('train', clip_len)
        else:
            self.clip_transform = util.clip_transform_ucf101('val', clip_len)

    def sample_rgb(self, entray):
        imgs = entray['frame']
        if len(imgs) > self.clip_len:

            if self.train:  # random sample
                offset = np.random.randint(0, len(imgs) - self.clip_len)
                imgs = imgs[offset:offset + self.clip_len]
            else:  # center crop
                offset = len(imgs) // 2 - self.clip_len // 2
                imgs = imgs[offset:offset + self.clip_len]
            assert len(imgs) == self.clip_len, 'frame selection error!'
        else:
            raise RuntimeError("len(imgs) > self.clip_len")

        imgs = [self.loader(img) for img in imgs]
        return imgs

    def sample_flow(self, entray):
        flow_x = entray['flow_x']
        flow_y = entray['flow_y']

        if len(flow_x) > self.clip_len:

            if self.train:  # random sample
                offset = np.random.randint(0, len(flow_x) - self.clip_len)
                flow_x = flow_x[offset:offset + self.clip_len]
                flow_y = flow_y[offset:offset + self.clip_len]
            else:  # center crop
                offset = len(flow_x) // 2 - self.clip_len // 2
                flow_x = flow_x[offset:offset + self.clip_len]
                flow_y = flow_y[offset:offset + self.clip_len]
            assert len(flow_x) == self.clip_len, 'frame selection error!'
        else:
            raise RuntimeError("len(imgs) > self.clip_len")

        flow_x_imgs = [self.loader(img) for img in flow_x]
        flow_y_imgs = [self.loader(img) for img in flow_y]

        return flow_x_imgs, flow_y_imgs

    def sample_all(self, entry):
        '''
        用于对多模态数据对进行采样
        :param entry: 包含多模态数据对的dict
        :return:
        '''
        rgb_img = entry['frame']
        flow_x = entry['flow_x']
        flow_y = entry['flow_y']

        if len(flow_x) > self.clip_len:

            if self.train:  # random sample
                offset = np.random.randint(0, len(flow_x) - self.clip_len)
                rgb_img = rgb_img[offset:offset + self.clip_len]
                flow_x = flow_x[offset:offset + self.clip_len]
                flow_y = flow_y[offset:offset + self.clip_len]
            else:  # center crop
                offset = len(flow_x) // 2 - self.clip_len // 2
                rgb_img = rgb_img[offset:offset + self.clip_len]
                flow_x = flow_x[offset:offset + self.clip_len]
                flow_y = flow_y[offset:offset + self.clip_len]
            assert len(flow_x) == self.clip_len, 'frame selection error!'
        else:
            raise RuntimeError("len(imgs) > self.clip_len")

        rgb_imgs = [self.loader(img) for img in rgb_img]
        flow_x_imgs = [self.loader(img) for img in flow_x]
        flow_y_imgs = [self.loader(img) for img in flow_y]

        return rgb_imgs, flow_x_imgs, flow_y_imgs

    def __getitem__(self, index):
        entry = self.data[index]
        if self.mode == 'rgb':
            rgb_frames = self.sample_rgb(entry)
            rgb_frames = self.clip_transform(rgb_frames)  # (T, 3, 224, 224)
            rgb_frames = rgb_frames.permute(1, 0, 2, 3)  # (3, T, 224, 224)
            b = datetime.datetime.now()
            # print((b-c).total_seconds())
            # print(entry['label'])
            instance = {'frames': rgb_frames, 'label': entry['label']}
        elif self.mode == 'flow':
            flow_x, flow_y = self.sample_flow(entry)
            # print(len(flow_x),len(flow_y))
            input = {"flow_x": flow_x, "flow_y": flow_y}
            output = self.clip_transform(input)
            flow_x, flow_y = output["flow_x"], output["flow_y"]
            zero_z = torch.zeros_like(flow_x)
            flow_xy = torch.cat((flow_x, flow_y), dim=1)
            flow_xy = flow_xy.permute(1, 0, 2, 3)
            instance = {'flow': flow_xy, 'label': entry['label']}
        elif self.mode == 'all':
            rgb_frames, flow_x, flow_y = self.sample_all(entry)
            input = {"rgb": rgb_frames, "flow_x": flow_x, "flow_y": flow_y}
            output = self.clip_transform(input)
            rgb_frames, flow_x, flow_y = output["rgb"], output["flow_x"], output["flow_y"]
            flow_xy = torch.cat((flow_x, flow_y), dim=1)
            flow_xy = flow_xy.permute(1, 0, 2, 3)
            rgb_frames = rgb_frames.permute(1, 0, 2, 3)
            instance = {'frames': rgb_frames, 'flow': flow_xy, 'label': entry['label']}
        else:
            raise RuntimeError("self.mode=='rgb'")

        return instance

    def __len__(self):
        # print(len(self.data))
        return len(self.data)

class UCF101_Multicrop(UCF101):
    def __init__(self, data_root, split_path, clip_len, mode, sample_interal=2):
        '''
        动作的label直接从文件夹的名字中提取:a01_s01_e01   a01 的01
        :param data_root: 存放NW-HMDB51 数据集的路径
        :param split_path: path for split txt
        :param clip_len: 每个动作取多少帧用于训练
        :mode : rgb or depth
        :param train: 训练还是测试
        :param sample_interal: 数据抽样加快训练
        '''
        super(UCF101_Multicrop, self).__init__(data_root, split_path, clip_len, mode, sample_interal=2)

        self.clip_transform = util.clip_transform_ucf101('3crop', self.clip_len)


    def sample_rgb(self, entray, K=10):
        imgs = entray['frame']
        # memoize loading images since clips overlap
        cache = {}

        def load(img):
            if img not in cache:
                cache[img] = self.loader(img)
            return cache[img]

        centers = [int(idx) for idx in np.linspace(self.clip_len // 2, len(imgs) - self.clip_len // 2, K)]

        clips = []

        for c in centers:

            clip = imgs[c:c + self.clip_len]
            clip = [load(img) for img in clip]
            clips.append(clip)

        return clips

    def sample_flow(self, entray,K=10):
        flow_x = entray['flow_x']
        flow_y = entray['flow_y']
        # memoize loading images since clips overlap
        cache = {}

        def load(img):
            if img not in cache:
                cache[img] = self.loader(img)
            return cache[img]

        centers = [int(idx) for idx in np.linspace(self.clip_len // 2, len(flow_x) - self.clip_len // 2, K)]

        clips = []

        for c in centers:
            flow_x_clip = flow_x[c:c + self.clip_len]
            flow_x_clip_load = [load(img) for img in flow_x_clip]

            flow_y_clip = flow_y[c:c + self.clip_len]
            flow_y_clip_load = [load(img) for img in flow_y_clip]

            clips.append({"flow_x": flow_x_clip_load, "flow_y": flow_y_clip_load})

            # clip = imgs[c - self.clip_len // 2:c + self.clip_len // 2]
            # clip = [load(img) for img in clip]
            # clips.append(clip)
        return clips

    def sample_all(self, entry,K=10):
        '''
        用于对多模态数据对进行采样
        :param entry: 包含多模态数据对的dict
        :return:
        '''
        rgb_img = entry['frame']
        flow_x = entry['flow_x']
        flow_y = entry['flow_y']
        cache = {}

        def load(img):
            if img not in cache:
                cache[img] = self.loader(img)
            return cache[img]

        centers = [int(idx) for idx in np.linspace(self.clip_len // 2, len(flow_x) - self.clip_len // 2, K)]

        clips = []
        for c in centers:
            flow_x_clip = flow_x[c:c + self.clip_len]
            flow_x_clip_load = [load(img) for img in flow_x_clip]

            flow_y_clip = flow_y[c:c + self.clip_len]
            flow_y_clip_load = [load(img) for img in flow_y_clip]

            rgb_frames = rgb_img[c:c + self.clip_len]
            rgb_frames_load = [load(img) for img in rgb_frames]

            clips.append({"rgb": rgb_frames_load, "flow_x": flow_x_clip_load, "flow_y": flow_y_clip_load})
        return clips

    def __getitem__(self, index):
        if self.mode == 'rgb':
            entry = self.data[index]
            clips = self.sample_rgb(entry)

            frames = []
            for clip in clips:
                clip = [self.clip_transform(clip).permute(1, 0, 2, 3) for _ in range(1)]
                clip = torch.stack(clip, 0)  # (3, 3, 224, 224)
                frames.append(clip)
            frames = torch.stack(frames, 0)  # (10, 3, 3, 224, 224)

            instance = {'frames': frames, 'label': entry['label']}
        elif self.mode=='flow':
            entry = self.data[index]
            clips = self.sample_flow(entry)

            frames = []

            for clip in clips:
                flow_xy_3 = []
                for i in range(3):
                    output = self.clip_transform(clip)
                    flow_x, flow_y = output["flow_x"], output["flow_y"]
                    zero_z = torch.zeros_like(flow_x)
                    flow_xy = torch.cat((flow_x, flow_y), dim=1)
                    flow_xy = flow_xy.permute(1, 0, 2, 3)
                    flow_xy_3.append(flow_xy)
                flow_xy_3 = torch.stack(flow_xy_3, dim=0)

                frames.append(flow_xy_3)
            frames = torch.stack(frames, 0)
            instance = {'flow': frames, 'label': entry['label']}
        elif self.mode=='all':
            entry = self.data[index]
            clips = self.sample_all(entry)

            rgb_frames = []
            flow_frames = []

            for clip in clips:
                flow_xy_3 = []
                rgb_3 = []
                for i in range(3):
                    output = self.clip_transform(clip)
                    rgb, flow_x, flow_y = output["rgb"], output["flow_x"], output["flow_y"]
                    zero_z = torch.zeros_like(flow_x)
                    flow_xy = torch.cat((flow_x, flow_y), dim=1)
                    flow_xy = flow_xy.permute(1, 0, 2, 3)
                    flow_xy_3.append(flow_xy)
                    rgb = rgb.permute(1, 0, 2, 3)
                    rgb_3.append(rgb)
                flow_xy_3 = torch.stack(flow_xy_3, dim=0)
                rgb_3 = torch.stack(rgb_3, dim=0)

                rgb_frames.append(rgb_3)
                flow_frames.append(flow_xy_3)
            rgb_frames = torch.stack(rgb_frames, 0)
            flow_frames = torch.stack(flow_frames, 0)
            instance = {'frames': rgb_frames, 'flow': flow_frames, 'label': entry['label']}
        else:
            print("self.mode==all")

        return instance

    def __len__(self):
        # print(len(self.data))
        return len(self.data)
