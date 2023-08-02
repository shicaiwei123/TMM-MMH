from torch.utils.data import Dataset

from lib.processing_utils import read_txt, replace_string
import glob
from PIL import Image
import numpy as np
from action_utils import util
import datetime
import torch
import os


class NWUCLA_CV(Dataset):
    def __init__(self, data_root, split_num, clip_len, mode, train=True, sample_interal=2):
        '''
        动作的label直接从文件夹的名字中提取:a01_s01_e01   a01 的01
        :param data_root: 存放NW-UCLA 数据集的路径
        :param split_num: 哪一个角度用于测试
        :param clip_len: 每个动作取多少帧用于训练
        :mode : rgb or depth
        :param train: 训练还是测试
        :param sample_interal: 数据抽样加快训练
        '''
        super(NWUCLA_CV, self).__init__()
        self.train_data = []
        self.test_data = []
        self.clip_len = clip_len

        view_list = [1, 2, 3]
        view_list.remove(split_num)
        train_view = view_list

        self.loader = lambda fl: Image.open(fl).convert('RGB')

        frames_path_len_min = 100000

        if train:
            for view in train_view:
                self.train_action_list = os.listdir(os.path.join(data_root, "view_" + str(view)))
                train_action_list_len = len(self.train_action_list)
                print(train_action_list_len)
                for i in range(train_action_list_len):
                    train_action = self.train_action_list[i]
                    action_label = train_action.split('_')[0]
                    action_label = int(action_label[1:len(action_label)]) - 1  # 减一 从0开始
                    if action_label > 9:  # (10,11)
                        continue

                    action_dir = os.path.join(data_root, "view_" + str(view), train_action)

                    # print("%s/%s" % (data_root, action_related_path))
                    if mode == 'rgb':
                        frames_path = glob.glob('%s/*rgb.jpg' % (action_dir))
                    elif mode == 'depth':
                        frames_path = glob.glob('%s/*depth_vis.jpg' % (action_dir))
                    else:
                        frames_path = []
                        raise ValueError("mode should be rgb or depth")
                    frames_path.sort()

                    if len(frames_path) < self.clip_len:
                        frames_path_len_min = len(frames_path)
                        # print(action_dir)

                    self.train_data.append({'frames': frames_path, "label": action_label})

                print(len(self.train_data))

        else:
            self.test_action_list = os.listdir(os.path.join(data_root, "view_" + str(split_num)))
            test_action_list_len = len(self.test_action_list)
            for i in range(test_action_list_len):
                test_action = self.test_action_list[i]
                action_label = test_action.split('_')[0]
                action_label = int(action_label[1:len(action_label)]) - 1  # 减一 从0开始
                if action_label > 9:  # (10,11)
                    continue
                action_dir = os.path.join(data_root, "view_" + str(split_num), test_action)

                # print("%s/%s" % (data_root, action_related_path))
                if mode == 'depth':
                    frames_path = glob.glob('%s/*rgb.jpg' % (action_dir))
                elif mode == 'rgb':
                    frames_path = glob.glob('%s/*depth_vis.jpg' % (action_dir))
                else:
                    frames_path = []
                    raise ValueError("mode should be rgb or depth")
                frames_path.sort()

                if len(frames_path) < self.clip_len:
                    frames_path_len_min = len(frames_path)
                    # print(action_dir)

                self.test_data.append({'frames': frames_path, "label": action_label})

            # print(len( self.test_data))

        if train:
            self.data = self.train_data
        else:
            self.data = self.test_data

        if train:
            self.clip_transform = util.clip_transform('train', clip_len)
        else:
            self.clip_transform = util.clip_transform('val', clip_len)

        self.train = train

        # print(frames_path_len_min)

    def sample(self, imgs):

        if len(imgs) > self.clip_len:

            if self.train:  # random sample
                offset = np.random.randint(0, len(imgs) - self.clip_len)
                imgs = imgs[offset:offset + self.clip_len]
            else:  # center crop
                offset = len(imgs) // 2 - self.clip_len // 2
                imgs = imgs[offset:offset + self.clip_len]
                assert len(imgs) == self.clip_len, 'frame selection error!'

        imgs = [self.loader(img) for img in imgs]
        return imgs

    def __getitem__(self, index):

        entry = self.data[index]
        # print(entry)
        a = datetime.datetime.now()
        frames = self.sample(entry['frames'])

        c = datetime.datetime.now()
        # print((c-a).total_seconds())

        frames = self.clip_transform(frames)  # (T, 3, 224, 224)
        frames = frames.permute(1, 0, 2, 3)  # (3, T, 224, 224)
        b = datetime.datetime.now()
        # print((b-c).total_seconds())
        # print(entry['label'])
        instance = {'frames': frames, 'label': entry['label']}

        return instance

    def __len__(self):
        # print(len(self.data))
        return len(self.data)


class NWUCLA_CS(Dataset):
    def __init__(self, data_root, split_subject, clip_len, mode, train=True, sample_interal=2):
        '''
        动作的label直接从文件夹的名字中提取:a01_s01_e01   a01 的01
        :param data_root: 存放NW-UCLA 数据集的路径
        :param split_num: 哪一个subject 用来测试
        :param clip_len: 每个动作取多少帧用于训练
        :mode : rgb or depth,or with all
        :param train: 训练还是测试
        :param sample_interal: 数据抽样加快训练
        '''
        super(NWUCLA_CS, self).__init__()
        self.train_data = []
        self.test_data = []
        self.clip_len = clip_len
        self.train = train
        self.mode = mode

        view_list = [i + 1 for i in range(10)]
        view_list.remove(split_subject)

        self.loader = lambda fl: Image.open(fl).convert('RGB')
        frames_path_len_min = 100000

        for view in [1, 2, 3]:
            self.view_action_path_list = glob.glob('%s/*' % (os.path.join(data_root, "view_" + str(view))))
            self.test_action_path_list = glob.glob(
                '%s/*_s%d_*' % (os.path.join(data_root, "view_" + str(view)), split_subject))
            self.train_action_path_list = list(set(self.view_action_path_list) - set(self.test_action_path_list))

            if train:
                for i in range(len(self.train_action_path_list)):
                    action_path = self.train_action_path_list[i]
                    action_label = action_path.split('/')[-1].split('_')[0]
                    action_label = int(action_label[1:len(action_label)]) - 1  # 减一 从0开始

                    if action_label > 9:  # (10,11)
                        continue

                    # print("%s/%s" % (data_root, action_related_path))
                    if mode == 'rgb':
                        frames_path = glob.glob('%s/*rgb.jpg' % (action_path))
                    elif mode == 'depth':
                        frames_path = glob.glob('%s/*depth_vis.jpg' % (action_path))
                    elif mode == 'all':
                        frames_path = []
                        frames_path_rgb = glob.glob('%s/*rgb.jpg' % (action_path))
                        frames_path_rgb.sort()
                        frames_path_depth = glob.glob('%s/*depth_vis.jpg' % (action_path))
                        frames_path_depth.sort()
                        self.train_data.append(
                            {'frames_rgb': frames_path_rgb, 'frames_depth': frames_path_depth, "label": action_label})
                    else:
                        frames_path = []
                        raise ValueError("mode should be rgb or depth")

                    if mode != 'all':
                        frames_path.sort()

                        if len(frames_path) < 10:
                            frames_path_len_min = len(frames_path)
                            print(action_path)

                        self.train_data.append({'frames': frames_path, "label": action_label})

            else:
                for i in range(len(self.test_action_path_list)):
                    action_path = self.test_action_path_list[i]
                    action_label = action_path.split('/')[-1].split('_')[0]
                    action_label = int(action_label[1:len(action_label)]) - 1  # 减一 从0开始

                    if action_label > 9:  # (10,11)
                        continue

                    # print("%s/%s" % (data_root, action_related_path))
                    if mode == 'rgb':
                        frames_path = glob.glob('%s/*rgb.jpg' % (action_path))
                    elif mode == 'depth':
                        frames_path = glob.glob('%s/*depth_vis.jpg' % (action_path))

                    elif mode == 'all':
                        frames_path = []
                        frames_path_rgb = glob.glob('%s/*rgb.jpg' % (action_path))
                        frames_path_rgb.sort()
                        frames_path_depth = glob.glob('%s/*depth_vis.jpg' % (action_path))
                        frames_path_depth.sort()
                        self.test_data.append(
                            {'frames_rgb': frames_path_rgb, 'frames_depth': frames_path_depth, "label": action_label})
                    else:
                        frames_path = []
                        raise ValueError("mode should be rgb or depth")

                    if mode != 'all':
                        frames_path.sort()

                        if len(frames_path) < 10:
                            frames_path_len_min = len(frames_path)
                            print(action_path)

                        self.test_data.append({'frames': frames_path, "label": action_label})

        print(len(self.train_data))
        print(len(self.test_data))

        if train:
            self.data = self.train_data
        else:
            self.data = self.test_data

        if train:
            self.clip_transform = util.clip_transform('train', clip_len)
        else:
            self.clip_transform = util.clip_transform('val', clip_len)

        # print(frames_path_len_min)

    def sample(self, imgs):

        if len(imgs) > self.clip_len:

            if self.train:  # random sample
                offset = np.random.randint(0, len(imgs) - self.clip_len)
                imgs = imgs[offset:offset + self.clip_len]
            else:  # center crop
                offset = len(imgs) // 2 - self.clip_len // 2
                imgs = imgs[offset:offset + self.clip_len]
                assert len(imgs) == self.clip_len, 'frame selection error!'

        imgs = [self.loader(img) for img in imgs]
        return imgs

    def sample_all(self, entry):
        '''
        用于对多模态数据对进行采样
        :param entry: 包含多模态数据对的dict
        :return:
        '''
        rgb_img_paths = entry['frames_rgb']
        depth_img_paths = entry['frames_depth']

        if len(rgb_img_paths) > self.clip_len:

            if self.train:  # random sample
                offset = np.random.randint(0, len(rgb_img_paths) - self.clip_len)
                rgb_img_paths = rgb_img_paths[offset:offset + self.clip_len]
                depth_img_paths = depth_img_paths[offset:offset + self.clip_len]
            else:  # center crop
                offset = len(rgb_img_paths) // 2 - self.clip_len // 2
                rgb_img_paths = rgb_img_paths[offset:offset + self.clip_len]
                depth_img_paths = depth_img_paths[offset:offset + self.clip_len]

                assert len(rgb_img_paths) == self.clip_len, 'frame selection error!'

        rgb_img_paths = [self.loader(img) for img in rgb_img_paths]
        depth_img_paths = [self.loader(img) for img in depth_img_paths]
        return rgb_img_paths, depth_img_paths

    def __getitem__(self, index):

        entry = self.data[index]
        if self.mode != 'all':
            # print(entry)
            a = datetime.datetime.now()
            frames = self.sample(entry['frames'])

            c = datetime.datetime.now()
            # print((c-a).total_seconds())

            frames = self.clip_transform(frames)  # (T, 3, 224, 224)
            frames = frames.permute(1, 0, 2, 3)  # (3, T, 224, 224)
            b = datetime.datetime.now()
            # print((b-c).total_seconds())
            # print(entry['label'])
            instance = {'frames': frames, 'label': entry['label']}
        else:
            self.clip_transform = util.clip_transform('train', self.clip_len)
            rgb_img_paths, depth_img_paths = self.sample_all(entry)
            rgb_img_paths = self.clip_transform(rgb_img_paths)  # (T, 3, 224, 224)
            rgb_img_paths = rgb_img_paths.permute(1, 0, 2, 3)  # (3, T, 224, 224)

            depth_img_paths = self.clip_transform(depth_img_paths)  # (T, 3, 224, 224)
            depth_img_paths = depth_img_paths.permute(1, 0, 2, 3)  # (3, T, 224, 224)

            instance = {'frames_rgb': rgb_img_paths, 'frames_depth': depth_img_paths, 'label': entry['label']}

        return instance

    def __len__(self):
        # print(len(self.data))
        return len(self.data)


class NWUCLACV_MultiCrop(NWUCLA_CV):

    def __init__(self, data_root, split_num, clip_len, mode, train=True, sample_interal=2):
        super(NWUCLACV_MultiCrop, self).__init__(data_root, split_num, clip_len, mode, train=train,
                                                 sample_interal=sample_interal)
        self.clip_transform = util.clip_transform('3crop', self.clip_len)

    def sample(self, imgs, K=10):
        '''
        每一个动作对应的一个视频的所有帧的路径
        :param imgs:
        :param K:
        :return:
        '''

        # memoize loading images since clips overlap
        cache = {}

        def load(img):
            if img not in cache:
                cache[img] = self.loader(img)
            return cache[img]

        centers = [int(idx) for idx in np.linspace(self.clip_len // 2, len(imgs) - self.clip_len // 2, K)]

        clips = []
        for c in centers:
            clip = imgs[c - self.clip_len // 2:c + self.clip_len // 2]
            clip = [load(img) for img in clip]
            clips.append(clip)
        return clips

    def __getitem__(self, index):

        entry = self.data[index]
        clips = self.sample(entry['frames'])

        frames = []
        for clip in clips:
            clip = [self.clip_transform(clip).permute(1, 0, 2, 3) for _ in range(3)]
            clip = torch.stack(clip, 0)  # (3, 3, 224, 224)
            frames.append(clip)
        frames = torch.stack(frames, 0)  # (10, 3, 3, 224, 224)

        instance = {'frames': frames, 'label': entry['label']}

        return instance


class NWUCLACS_MultiCrop(NWUCLA_CS):

    def __init__(self, data_root, split_subject, clip_len, mode, train=True, sample_interal=2):
        super(NWUCLACS_MultiCrop, self).__init__(data_root, split_subject, clip_len, mode, train=train,
                                                 sample_interal=sample_interal)
        self.clip_transform = util.clip_transform('3crop', self.clip_len)

    def sample(self, imgs, K=10):
        '''
        每一个动作对应的一个视频的所有帧的路径
        :param imgs:
        :param K:
        :return:
        '''

        # memoize loading images since clips overlap
        cache = {}

        def load(img):
            if img not in cache:
                cache[img] = self.loader(img)
            return cache[img]

        centers = [int(idx) for idx in np.linspace(self.clip_len // 2, len(imgs) - self.clip_len // 2, K)]

        clips = []
        for c in centers:
            clip = imgs[c - self.clip_len // 2:c + self.clip_len // 2]
            clip = [load(img) for img in clip]
            clips.append(clip)
        return clips

    def sample_all(self, entry, K=10):
        '''
        用于对多模态数据对进行采样
        :param entry: 包含多模态数据对的dict
        :return:
        '''
        rgb_img_paths = entry['frames_rgb']
        depth_img_paths = entry['frames_depth']

        cache = {}

        def load(img):
            if img not in cache:
                cache[img] = self.loader(img)
            return cache[img]

        centers = [int(idx) for idx in np.linspace(self.clip_len // 2, len(rgb_img_paths) - self.clip_len // 2, K)]

        rgb_clips = []
        depth_clips = []
        for c in centers:
            rgb_clip = rgb_img_paths[c - self.clip_len // 2:c + self.clip_len // 2]
            rgb_clip = [load(img) for img in rgb_clip]
            rgb_clips.append(rgb_clip)

            depth_clip = depth_img_paths[c - self.clip_len // 2:c + self.clip_len // 2]
            depth_clip = [load(img) for img in depth_clip]
            depth_clips.append(depth_clip)

        return rgb_clips, depth_clips

    def __getitem__(self, index):

        if self.mode != 'all':

            entry = self.data[index]
            clips = self.sample(entry['frames'])

            frames = []
            for clip in clips:
                clip = [self.clip_transform(clip).permute(1, 0, 2, 3) for _ in range(3)]
                clip = torch.stack(clip, 0)  # (3, 3, 224, 224)
                frames.append(clip)
            frames = torch.stack(frames, 0)  # (10, 3, 3, 224, 224)

            instance = {'frames': frames, 'label': entry['label']}

        else:
            self.clip_transform = util.clip_transform('val', self.clip_len)

            entry = self.data[index]
            rgb_clips, depth_clips = self.sample_all(entry)

            frames_rgb = []
            for i in range(len(rgb_clips)):
                clip = rgb_clips[i]
                clip = [self.clip_transform(clip).permute(1, 0, 2, 3) for _ in range(3)]
                clip = torch.stack(clip, 0)  # (3, 3, 224, 224)
                frames_rgb.append(clip)
            frames_rgb = torch.stack(frames_rgb, 0)  # (10, 3, 3, 224, 224)

            frames_depth = []
            for i in range(len(depth_clips)):
                clip = depth_clips[i]
                clip = [self.clip_transform(clip).permute(1, 0, 2, 3) for _ in range(3)]
                clip = torch.stack(clip, 0)  # (3, 3, 224, 224)
                frames_depth.append(clip)
            frames_depth = torch.stack(frames_depth, 0)  # (10, 3, 3, 224, 224)

            instance = {'frames_rgb': frames_rgb, 'frames_depth': frames_depth, 'label': entry['label']}

        return instance
