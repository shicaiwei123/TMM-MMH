import types
import collections
import numpy as np
from random import shuffle
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import os
from lib.processing_utils import read_txt
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline

batch_size = 32


class ExternalInputIterator(object):
    def __init__(self, data_root, split_path, clip_len, mode, batch_size, ):
        self.batch_size = batch_size

        self.test_data = []
        self.train_data = []

        self.clip_len = clip_len
        self.mode = mode

        txt_name = split_path.split('/')[-1]
        if txt_name.split('_')[0] == 'train':
            self.train = True
        elif txt_name.split('_')[0] == 'test':
            self.train = False
        else:
            print('error file')

        split_line = read_txt(split_path)
        for line in split_line:
            related_path = line.split(' ')[0]
            related_path = related_path.split('.')[0]
            label = int(line.split(' ')[1])
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

        if self.train:
            self.files = self.train_data
            shuffle(self.files)
        else:
            self.files = self.test_data

        self.data_set_len = len(self.files)

    def __iter__(self):
        self.i = 0
        self.n = len(self.files)
        return self

    def __next__(self):
        batch = []
        for _ in range(self.batch_size):
            entry = self.files[self.i]
            if self.mode == 'rgb':
                rgb_frames = self.sample_rgb(entry)
                instance = {'frames': rgb_frames, 'label': entry['label']}
            elif self.mode == 'flow':
                flow_x, flow_y = self.sample_flow(entry)
            elif self.mode == 'all':
                print("to do")
            else:
                raise RuntimeError("self.mode=='rgb'")
            batch.append(instance)
            self.i = (self.i + 1) % self.n
        return batch

    @property
    def size(self, ):
        return self.data_set_len

    next = __next__

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

        # imgs = [self.loader(img) for img in imgs]
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
                flow_y = flow_x[offset:offset + self.clip_len]
            assert len(flow_x) == self.clip_len, 'frame selection error!'
        else:
            raise RuntimeError("len(imgs) > self.clip_len")

        # flow_x_imgs = [self.loader(img) for img in flow_x]
        # flow_y_imgs = [self.loader(img) for img in flow_y]

        return flow_x, flow_y

    def sample_all(self, entry):
        '''
        用于对多模态数据对进行采样
        :param entry: 包含多模态数据对的dict
        :return:
        '''
        rgb_img = entry['frames']
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
                flow_y = flow_x[offset:offset + self.clip_len]
            assert len(flow_x) == self.clip_len, 'frame selection error!'
        else:
            raise RuntimeError("len(imgs) > self.clip_len")

        # rgb_imgs = [self.loader(img) for img in rgb_img]
        # flow_x_imgs = [self.loader(img) for img in flow_x]
        # flow_y_imgs = [self.loader(img) for img in flow_y]

        return rgb_img, flow_x, flow_y


class ExternalSourcePipeline(Pipeline):
    def __init__(self, resize, batch_size, num_threads, device_id, external_data):
        super(ExternalSourcePipeline, self).__init__(batch_size,
                                                     num_threads,
                                                     device_id,
                                                     seed=12,
                                                     exec_async=False,
                                                     exec_pipelined=False,
                                                     )

        self.external_data = external_data  # ExternalInputIterator
        self.iterator = iter(self.external_data)

        self.inputs = ops.ExternalSource()
        self.input_label = ops.ExternalSource()

        self.decode = ops.ImageDecoderCrop(device = 'gpu', output_type = types.RGB, crop = (112, 112))



        def build_transform(self):
            res = []
            size_train = self.cfg.DATA.INPUT_SIZE
            resize = ops.Resize(device="gpu", resize_x=size_train[0], resize_y=size_train[1],
                                interp_type=types.INTERP_TRIANGULAR)
            res.append(resize)
            return res

        def define_graph(self):
            # 定义图，数据解码、transform 都在这里
            batch_data = [i() for i in self.inputs]
            self.images = batch_data
            self.labels = self.input_label()

        out = fn.decoders.image(self.images, device="mixed", output_type=types.RGB)
        out = [out_elem.gpu() for out_elem in out]
        for trans in self.transforms:
            out = trans(out)
            # 注意，这里是 *out ，即最后拿到的是一个 list
            # 所以 iter_setup 里也是循环 n 次进行 feed_input
            # 把 images 序列有 list 方式弹出；
            return (*out, self.labels)

        def iter_setup(self):
            try:
                batch_data, labels = self.iterator.next()  # 拿到一个Batch的数据，对应上面的 ExternalInputIterator 的 next 拿到的结果
                # batch_data 中的每个元素都构建一个 feed_input，每个feed_input 操作 batch 个数据
                for i in range(self.images_length):
                    self.feed_input(self.images[i], batch_data[i])
                self.feed_input(self.labels, labels)
            except StopIteration:
                self.iterator = iter(self.external_data)
                raise StopIteration
