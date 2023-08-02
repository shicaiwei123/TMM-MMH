import sys

sys.path.append('..')

from lib.processing_utils import read_txt, save_csv, replace_string
from lib.processing_utils import seed_torch
import torch
import numpy as np
import torch.nn.functional as F
import cv2
import random
import torchvision
import torchvision.transforms.transforms as ttf
import csv
import os

'''对图像transform的测试'''

# img=cv2.imread("/home/shicaiwei/data/liveness_data/CASIA-SUFR/Training/real_part/CLKJ_AS0137/real.rssdk/depth/231.jpg")
# img=cv2.resize(img,(112,112))
# h, w = img.shape[0], img.shape[1]  # Tensor [1][2],  nparray [0][1]
# length_new = np.random.randint(1, 30)
# print("abc",length_new)
# y = np.random.randint(h - length_new)
# x = np.random.randint(w - length_new)
#
# img[y:y + length_new, x:x + length_new] = 0
# cv2.imshow("img",img)
# cv2.waitKey(0)
#
#

# import torch as t
# import numpy as np
#
# batch_size = 5
# class_num = 2
# label = np.random.randint(0, class_num, size=(batch_size, 1))
# label = t.LongTensor(label)
# y_one_hot = t.zeros(batch_size, class_num).scatter_(1, label, 1)
# print(y_one_hot)
#
# # a = torch.tensor([[0], [1], [2], [3], [4]]).long()
# # b = torch.zeros(5, 2)
# # c = b.scatter_(1, a, 1)
# # print(1)
# img = Image.open(
#     "/home/shicaiwei/data/liveness_data/CASIA-SUFR/Training/real_part/CLKJ_AS0137/real.rssdk/depth/31.jpg").convert(
#     'RGB')
# img2 = cv2.imread(
#     "/home/shicaiwei/data/liveness_data/CASIA-SUFR/Training/real_part/CLKJ_AS0137/real.rssdk/depth/31.jpg")
# img2 = np.float32(img2)
# img2 = torch.from_numpy(img2)
# img1 = img.resize((32, 32))
#
# img2 = F.adaptive_avg_pool2d(img2, (32, 32))
# img2 = np.uint8(np.array(img2))
# img1.show()
# cv2.imshow("img2", img2)
# cv2.waitKey(0)

from lib.processing_utils import read_txt


def generate_single_list(related_sample_path_list):
    fw = open('/home/shicaiwei/data/liveness_data/CASIA-SUFR/test_private_single_list.txt', 'a+')
    for idx in range(len(related_sample_path_list)):
        related_sample_path = related_sample_path_list[idx]
        related_sample_path_split = related_sample_path.split(" ")
        related_sample_path_split[0] = 'color/' + related_sample_path_split[0]
        related_sample_path_split[1] = 'depth/' + related_sample_path_split[1]
        related_sample_path_split[2] = 'ir/' + related_sample_path_split[2]
        binary_label = np.int(related_sample_path_split[3])
        if binary_label == 1:
            related_sample_path_split_0_split = related_sample_path_split[0].split('/')
            related_sample_path_split_0_split[1] = related_sample_path_split_0_split[1] + '/real_park'
            related_sample_path_split[0] = '/'.join(related_sample_path_split_0_split)

            related_sample_path_split_1_split = related_sample_path_split[1].split('/')
            related_sample_path_split_1_split[1] = related_sample_path_split_1_split[1] + '/real_park'
            related_sample_path_split[1] = '/'.join(related_sample_path_split_1_split)

            related_sample_path_split_2_split = related_sample_path_split[2].split('/')
            related_sample_path_split_2_split[1] = related_sample_path_split_2_split[1] + '/real_park'
            related_sample_path_split[2] = '/'.join(related_sample_path_split_2_split)
        elif binary_label == 0:
            related_sample_path_split_0_split = related_sample_path_split[0].split('/')
            related_sample_path_split_0_split[1] = related_sample_path_split_0_split[1] + '/fake_park'
            related_sample_path_split[0] = '/'.join(related_sample_path_split_0_split)

            related_sample_path_split_1_split = related_sample_path_split[1].split('/')
            related_sample_path_split_1_split[1] = related_sample_path_split_1_split[1] + '/fake_park'
            related_sample_path_split[1] = '/'.join(related_sample_path_split_1_split)

            related_sample_path_split_2_split = related_sample_path_split[2].split('/')
            related_sample_path_split_2_split[1] = related_sample_path_split_2_split[1] + '/fake_park'
            related_sample_path_split[2] = '/'.join(related_sample_path_split_2_split)
        related_sample_path = ' '.join(related_sample_path_split)
        fw.write(str(related_sample_path))
        fw.write('\n')

    fw.close()


def generate_gan_list(related_sample_path_list):
    fw = open('/home/shicaiwei/data/liveness_data/CASIA-SUFR/test_private_gan_list.txt', 'a+')
    for idx in range(len(related_sample_path_list)):
        related_sample_path = related_sample_path_list[idx]
        related_sample_path_split = related_sample_path.split(" ")
        related_sample_path_split[0] = 'color/' + related_sample_path_split[0]
        related_sample_path_split[1] = 'color2depth/' + related_sample_path_split[1]
        related_sample_path_split[2] = 'color2ir/' + related_sample_path_split[2]
        binary_label = np.int(related_sample_path_split[3])
        if binary_label == 1:
            related_sample_path_split_0_split = related_sample_path_split[0].split('/')
            related_sample_path_split_0_split[1] = related_sample_path_split_0_split[1] + '/real_park'
            related_sample_path_split[0] = '/'.join(related_sample_path_split_0_split)

            related_sample_path_split_1_split = related_sample_path_split[1].split('/')
            related_sample_path_split_1_split[2] = '/real_park'
            related_sample_path_split_1_split_split = related_sample_path_split_1_split[-1].split('-')
            related_sample_path_split_1_split_split[-1] = 'color_fake.png'
            related_sample_path_split_1_split[-1] = '-'.join(related_sample_path_split_1_split_split)
            related_sample_path_split[1] = '/'.join(related_sample_path_split_1_split)

            related_sample_path_split_2_split = related_sample_path_split[2].split('/')
            related_sample_path_split_2_split[2] = '/real_park'
            related_sample_path_split_2_split_split = related_sample_path_split_2_split[-1].split('-')
            related_sample_path_split_2_split_split[-1] = 'color_fake.png'
            related_sample_path_split_2_split[-1] = '-'.join(related_sample_path_split_2_split_split)
            related_sample_path_split[2] = '/'.join(related_sample_path_split_2_split)

        elif binary_label == 0:
            #
            related_sample_path_split_0_split = related_sample_path_split[0].split('/')
            related_sample_path_split_0_split[1] = related_sample_path_split_0_split[1] + '/fake_park'
            related_sample_path_split[0] = '/'.join(related_sample_path_split_0_split)

            #
            related_sample_path_split_1_split = related_sample_path_split[1].split('/')
            related_sample_path_split_1_split[2] = '/fake_park'

            related_sample_path_split_1_split_split = related_sample_path_split_1_split[-1].split('-')
            related_sample_path_split_1_split_split[-1] = 'color_fake.png'
            related_sample_path_split_1_split[-1] = '-'.join(related_sample_path_split_1_split_split)

            related_sample_path_split[1] = '/'.join(related_sample_path_split_1_split)

            #
            related_sample_path_split_2_split = related_sample_path_split[2].split('/')
            related_sample_path_split_2_split[2] = '/fake_park'

            related_sample_path_split_2_split_split = related_sample_path_split_2_split[-1].split('-')
            related_sample_path_split_2_split_split[-1] = 'color_fake.png'
            related_sample_path_split_2_split[-1] = '-'.join(related_sample_path_split_2_split_split)

            related_sample_path_split[2] = '/'.join(related_sample_path_split_2_split)
        related_sample_path = ' '.join(related_sample_path_split)
        fw.write(str(related_sample_path))
        fw.write('\n')

    fw.close()


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    将源域数据和目标域数据转化为核矩阵，即上文中的K
    Params:
	    source: 源域数据（n * len(x))
	    target: 目标域数据（m * len(y))
	    kernel_mul:
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		sum(kernel_val): 多个核矩阵之和
    '''

    n_samples = int(source.size()[0]) + int(target.size()[0])  # 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
    total = torch.cat([source, target], dim=0)  # 将source,target按列方向合并
    # 将total复制（n+m）份
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # 将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # 求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
    L2_distance = ((total0 - total1) ** 2).sum(2)
    # 调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    # 以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    # 高斯核函数的数学表达式
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    # 得到最终的核矩阵
    return sum(kernel_val)  # /len(kernel_val)


def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
	    source: 源域数据（n * len(x))
	    target: 目标域数据（m * len(y))
	    kernel_mul:
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		loss: MMD loss
    '''
    batch_size = int(source.size()[0])  # 一般默认为源域和目标域的batchsize相同
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    # 根据式（3）将核矩阵分成4部分
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss  # 因为一般都是n==m，所以L矩阵一般不加入计算


# from scipy.stats import wasserstein_distance
#
# x0 = wasserstein_distance([0, 1, 3], [0, 1, 3])
# data_1 = torch.tensor(np.random.normal(0, 10, (100, 50)))
# data_2 = torch.tensor(np.random.normal(10, 10, (100, 50)))
# a = torch.tensor([[0.1, 0.5, 0.5]])
# b = torch.tensor([[0.1, 0.4, 0.4]])
# c = tnf.binary_cross_entropy(a, b, weight=torch.tensor([1.0,2.0,1.0]), size_average=True)
# a_softmax = torch.softmax(a, dim=0)
# b_softmax = torch.softmax(b, dim=0)
# d = tnf.kl_div(a.log(), b)
# e = mmd_rbf(data_1, data_2)
# f = wasserstein_distance(a, b)
# h = tnf.mse_loss(a, b)
# print(c)

# import bagnets.pytorchnet
# from bagnets.utils import plot_heatmap, generate_heatmap_pytorch
#
# image_path = "/home/shicaiwei/data/liveness_data/cross_replayed_face/intra_test/living/1/0.jpg"
# im = cv2.imread(image_path)
# original = cv2.resize(im, (224, 224))
# original = np.transpose(original, (2, 0, 1))
#
# # preprocess sample image
# sample = original / 255.
# sample -= np.array([0.485, 0.456, 0.406])[:, None, None]
# sample /= np.array([0.229, 0.224, 0.225])[:, None, None]
# pytorch_model = bagnets.pytorchnet.bagnet17(pretrained=True)
# heatmap = generate_heatmap_pytorch(pytorch_model, sample, 1, 33)
#
# img = cv2.imread("/home/shicaiwei/data/liveness_data/cross_replayed_face/intra_test/living/1/0.jpg")

# a=torch.randn((10,2))
# from models.resnet_se_patch import resnet18_se_patch
# from configuration.config_patch_kd import args
# args.modal=args.student_modal
# a=resnet18_se_patch(args=args)
# a=a.conv1
# print(a)


def margin_plot():
    import numpy as np
    import matplotlib.pyplot as plt

    # fig=plt.figure()
    theta = np.linspace(0, 2 * np.pi, 200)
    x = np.cos(theta) * 0.2 + 0.5
    y = np.sin(theta) * 0.2 + 0.5
    # fig, ax = plt.subplots(figsize=(4, 4))
    # plt.show()

    logits_outputs = []
    labels = []
    # output_data_path = "surf_vanlia_prediction.txt"
    output_data_path = "surf_dad_prediction.txt"
    output_data = read_txt(output_data_path)
    for line in output_data:
        line = eval(line)
        logits_outputs.append(list(line[:-1]))
        labels.append(line[-1])

    logits_outputs = torch.tensor(logits_outputs).float()
    labels = torch.tensor(labels)
    logits_outputs = F.softmax(logits_outputs, dim=1)

    logits_cls = []
    fig = plt.figure()
    # ax = plt.subplot()
    for i in range(2):
        label_mask = labels == i
        logits_cls.append(logits_outputs[label_mask])

        sample = np.array(logits_cls[i])
        c = sample[:, 0]
        d = sample[:, 1]
        # class 0
        if i == 0:
            c = c[1:1000]
            d = d[1:1000]
            d = 1 - d  # fanzhuan
            true_mask = d > 0.4
            c = c[true_mask]
            d = d[true_mask]
            d = 1 - d
            index = [i for i in range(0, len(c), 3)]


        # class 1,
        else:
            c = c[1:1000]
            d = d[1:1000]
            d = 1 - d
            true_mask = d < 0.6
            c = c[true_mask]
            d = d[true_mask]
            d = 1 - d
            index = [i for i in range(0, len(c), 3)]

        # d = (d - 0.5) * 2
        c = c[index]
        d = d[index]
        plt.scatter(c, d, alpha=2 / 5)

        # true_mask = c > 0.4
        # c = c[true_mask]
        # d = d[true_mask]
        # if i == 0:
        #     c = 1 - c
        # # d = (d - 0.5) * 2
        # index = [i for i in range(0, len(c), 1)]
        # c = c[index]
        # d = d[index]
        # plt.scatter(c, d)

    # plt.xlim([-0.1, 1.1])
    # plt.axvline(0.5, c='black')
    plt.plot([0, 1], c='g', linestyle='--')
    plt.plot(x, y, color="g", linewidth=1, linestyle='--')

    # plt.ylim([-0.1, 1.1])
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    # plt.savefig('/home/shicaiwei/data/logits.pdf', dpi=1200, format='pdf')
    #
    plt.show()


def margin_plot_plant():
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import Counter

    # fig=plt.figure()
    theta = np.linspace(0, 2 * np.pi, 200)
    x = np.cos(theta) * 0.2 + 0.5
    y = np.sin(theta) * 0.2 + 0.5
    # fig, ax = plt.subplots(figsize=(4, 4))
    # plt.show()

    logits_outputs = []
    labels = []
    output_data_path = "surf_vanlia_prediction.txt"
    # output_data_path = "surf_dad_prediction.txt"
    # output_data_path = "logits_label.txt"
    output_data = read_txt(output_data_path)
    for line in output_data:
        line = eval(line)
        logits_outputs.append(list(line[:-1]))
        labels.append(line[-1])

    logits_outputs = torch.tensor(logits_outputs).float()
    labels = torch.tensor(labels)
    logits_outputs = F.softmax(logits_outputs, dim=1)

    logits_cls = []
    fig = plt.figure()
    # ax = plt.subplot()
    for i in range(2):
        label_mask = labels == i
        logits_cls.append(logits_outputs[label_mask])

        sample = np.array(logits_cls[i])
        c = sample[:, 0]
        d = sample[:, 1]
        # class 0
        if i == 0:
            c = c[1:2000]
            d = d[1:2000]
            d = 1 - d  # fanzhuan
            true_mask = d > 0.1
            c = c[true_mask]
            d = d[true_mask]
            d = 1 - d
            index = [i for i in range(0, len(c), 2)]


        # class 1,
        else:
            c = c[1:2000]
            d = d[1:2000]
            d = 1 - d
            true_mask = d < 0.9
            c = c[true_mask]
            d = d[true_mask]
            d = 1 - d
            index = [i for i in range(0, len(c), 2)]

        # d = (d - 0.5) * 2
        c = c[index]
        d = d[index]
        c = np.array(c)
        c = np.around(c, 2)
        d = np.array(d)
        d = np.around(d, 2)
        c_hist = Counter(c)
        x = []
        y = []
        # d_hist=Counter(d)
        # plt.plot(c_hist)
        #
        # for i in range(100):
        #     c_select = c[c >= (0.01) * i]
        #     c_select = c_select[c_select < 0.01 * (i + 1)]
        #     c_select_len=len(c_select)
        #     for i in range(c_select_len):
        #         x.append(0)
        #
        # plt.scatter(c, d)
        for k, v in c_hist.items():
            for i in range(v):
                ccc = np.random.rand(1)
                ccc = np.around(ccc, 2)
                x.append(k)
                y.append(i * ccc)

        plt.scatter(x, y)

        # true_mask = c > 0.4
        # c = c[true_mask]
        # d = d[true_mask]
        # if i == 0:
        #     c = 1 - c
        # # d = (d - 0.5) * 2
        # index = [i for i in range(0, len(c), 1)]
        # c = c[index]
        # d = d[index]
        # plt.scatter(c, d)

        plt.axvline(0.5, linestyle='--', color='black')

    plt.ylim([0.5, 5])
    # plt.xlim([0.01,0.99])
    # plt.axvline(0.5, c='black')
    # plt.plot([0, 1], c='g', linestyle='--')
    # plt.plot(x, y, color="g", linewidth=1, linestyle='--')

    # plt.ylim([-0.1, 1.1])
    plt.xticks(fontsize=11)
    plt.yticks([])
    # plt.savefig('/home/shicaiwei/data/rgb_multi_van_0.5_4.pdf', dpi=1200, format='pdf')

    plt.show()


def margin_plot_hist():
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import Counter

    # fig=plt.figure()
    theta = np.linspace(0, 2 * np.pi, 200)
    x = np.cos(theta) * 0.2 + 0.5
    y = np.sin(theta) * 0.2 + 0.5
    # fig, ax = plt.subplots(figsize=(4, 4))
    # plt.show()

    logits_outputs = []
    labels = []
    # output_data_path = "surf_vanlia_prediction.txt"
    # output_data_path = "surf_dad_prediction.txt"
    output_data_path = "logits_label.txt"
    output_data = read_txt(output_data_path)
    for line in output_data:
        line = eval(line)
        logits_outputs.append(list(line[:-1]))
        labels.append(line[-1])

    logits_outputs = torch.tensor(logits_outputs).float()
    labels = torch.tensor(labels)
    # logits_outputs = F.softmax(logits_outputs, dim=1)

    logits_cls = []
    fig = plt.figure()
    # ax = plt.subplot()
    for i in range(2):
        label_mask = labels == i
        logits_cls.append(logits_outputs[label_mask])

        sample = np.array(logits_cls[i])
        c = sample[:, 0]
        d = sample[:, 1]
        # class 0
        if i == 0:
            c = c[1:1500]
            d = d[1:1500]
            # d = 1 - d  # fanzhuan
            # true_mask = d > 0.1
            # c = c[true_mask]
            # d = d[true_mask]
            # d = 1 - d
            index = [i for i in range(0, len(c), 2)]


        # class 1,
        else:
            c = c[1:1500]
            d = d[1:1500]
            # d = 1 - d
            # true_mask = d < 0.9
            # c = c[true_mask]
            # d = d[true_mask]
            # d = 1 - d
            index = [i for i in range(0, len(c), 2)]

        # d = (d - 0.5) * 2
        c = c[index]
        d = d[index]
        c = np.array(c)
        c = np.around(c, 2)
        d = np.array(d)
        d = np.around(d, 2)
        c_hist = Counter(c)
        plt.hist(c, bins=50, alpha=0.8)

        # true_mask = c > 0.4
        # c = c[true_mask]
        # d = d[true_mask]
        # if i == 0:
        #     c = 1 - c
        # # d = (d - 0.5) * 2
        # index = [i for i in range(0, len(c), 1)]
        # c = c[index]
        # d = d[index]
        # plt.scatter(c, d)

        plt.axvline(0.0, linestyle='--', color='black')

    # plt.ylim([0.5, 3])
    # plt.xlim([0.01,0.99])
    # plt.axvline(0.5, c='black')
    # plt.plot([0, 1], c='g', linestyle='--')
    # plt.plot(x, y, color="g", linewidth=1, linestyle='--')

    # plt.ylim([-0.1, 1.1])
    plt.xticks(fontsize=11)
    plt.yticks([])
    # plt.savefig('/home/shicaiwei/data/rgb_van_distribution_soft.pdf', dpi=1200, format='pdf')

    plt.show()


if __name__ == '__main__':
    # target=read_txt("/home/shicaiwei/project/multimodal_inheritance/multi_model_fas/test/checksum.txt")
    # test=read_txt("/home/shicaiwei/project/multimodal_inheritance/multi_model_fas/test/log.txt")
    #
    # txt_len=len(target)
    # for i in range(txt_len):
    #     target_line=target[i].split(" ")[0].strip()
    #     test_line=test[i].split(" ")[0].strip()
    #     print(id(target_line),id(test_line))
    #     if test_line is target_line:
    #         print(1)
    #     for index in range(len(target_line)):
    #         a="aaa1"
    #         b="aaa1"
    #         print(id(a),id(b))
    #         if not (a == b):
    #             print(i)
    # a=torch.ones((128,2))
    # a=a.cpu().detach().numpy()
    # save_csv("surf_dad_prediction.csv",a)
    # print(1)
    # print(1)

    margin_plot_plant()

    # video_dir = "/home/data/shicaiwei/kinects400/rawframes_train/"
    # action_list = os.listdir(video_dir)
    # for video in action_list:
    #     action_dir = os.path.join(video_dir, video)
    #     action_video_list = os.listdir(action_dir)
    #     for aa in action_video_list:
    #         action_video_path = os.path.join(action_dir, aa)
    #         frame_file = os.listdir(action_video_path)
    #         if len(frame_file) > 400:
    #             video_path = action_video_path + ".mp4"
    #             video_path = replace_string(video_path, 5, 'train')
    #             if os.path.exists(video_path):
    #                 print(video_path)
    #                 os.remove(video_path)
    # print(action_video_path, len(frame_file))
    # _000001_000011.mp4
    # /home/data/shicaiwei/kinects400/train/passing_American_football_(not_in_game)/fbtcLtxD1VI_000003_000013.mp4
    # /home/data/shicaiwei/kinects400/train/passing_American_football_(not_in_game)/Jf8w7DpqWXM_000014_000024.mp4
    # /home/data/shicaiwei/kinects400/train/passing_American_football_(not_in_game)/sL52hYqB0qc_000007_000017.mp4
    # /home/data/shicaiwei/kinects400/train/passing_American_foot
    #     a=torch.tensor([[1,2,3,4],[5,6,7,8]]).float()
    #     b=torch.tensor([[2,3,4,4],[3,3,3,4]]).float()
    #     a=torch.log_softmax(a,dim=1)
    #     b=torch.softmax(b,dim=1)
    #     c=F.kl_div(a,b,reduction='batchmean')
    #     print(c)
    #     cc=F.kl_div(a,b,reduction='none')
    #     print(cc)
    #     print(torch.sum(cc,dim=1))

    dist = torch.tensor([[[0.0396, 0.0432, 0.0433, 0.0406, 0.0405, 0.0473, 0.0472, 0.0428,
                           0.0448, 0.0573, 0.0596, 0.0507, 0.0482, 0.0540, 0.0560, 0.0522,
                           0.0415, 0.0424, 0.0499, 0.0541, 0.0446],
                          [0.0284, 0.0358, 0.0359, 0.0304, 0.0302, 0.0454, 0.0451, 0.0348,
                           0.0392, 0.0749, 0.0831, 0.0542, 0.0477, 0.0640, 0.0704, 0.0588,
                           0.0322, 0.0341, 0.0521, 0.0644, 0.0389],
                          [0.0282, 0.0356, 0.0358, 0.0302, 0.0300, 0.0453, 0.0450, 0.0346,
                           0.0391, 0.0753, 0.0837, 0.0543, 0.0476, 0.0642, 0.0708, 0.0589,
                           0.0320, 0.0339, 0.0521, 0.0646, 0.0387],
                          [0.0361, 0.0411, 0.0412, 0.0375, 0.0373, 0.0470, 0.0468, 0.0404,
                           0.0432, 0.0622, 0.0660, 0.0519, 0.0483, 0.0570, 0.0601, 0.0543,
                           0.0387, 0.0400, 0.0508, 0.0572, 0.0430],
                          [0.0364, 0.0413, 0.0414, 0.0378, 0.0376, 0.0470, 0.0468, 0.0406,
                           0.0434, 0.0618, 0.0654, 0.0518, 0.0483, 0.0567, 0.0598, 0.0541,
                           0.0389, 0.0402, 0.0507, 0.0569, 0.0432],
                          [0.0193, 0.0281, 0.0284, 0.0216, 0.0214, 0.0416, 0.0412, 0.0269,
                           0.0327, 0.0947, 0.1124, 0.0558, 0.0451, 0.0732, 0.0857, 0.0637,
                           0.0237, 0.0260, 0.0523, 0.0740, 0.0323],
                          [0.0195, 0.0283, 0.0286, 0.0218, 0.0216, 0.0417, 0.0413, 0.0271,
                           0.0329, 0.0942, 0.1116, 0.0558, 0.0452, 0.0729, 0.0853, 0.0636,
                           0.0239, 0.0262, 0.0523, 0.0737, 0.0325],
                          [0.0296, 0.0366, 0.0368, 0.0315, 0.0314, 0.0457, 0.0454, 0.0357,
                           0.0399, 0.0727, 0.0801, 0.0539, 0.0478, 0.0628, 0.0687, 0.0581,
                           0.0332, 0.0350, 0.0520, 0.0632, 0.0396],
                          [0.0246, 0.0328, 0.0330, 0.0268, 0.0266, 0.0441, 0.0438, 0.0317,
                           0.0368, 0.0824, 0.0938, 0.0551, 0.0469, 0.0677, 0.0763, 0.0609,
                           0.0287, 0.0309, 0.0524, 0.0683, 0.0364],
                          [0.0075, 0.0149, 0.0152, 0.0092, 0.0090, 0.0305, 0.0299, 0.0138,
                           0.0197, 0.1363, 0.1861, 0.0519, 0.0353, 0.0851, 0.1135, 0.0661,
                           0.0109, 0.0129, 0.0461, 0.0868, 0.0192],
                          [0.0061, 0.0128, 0.0131, 0.0076, 0.0074, 0.0280, 0.0274, 0.0117,
                           0.0173, 0.1442, 0.2027, 0.0502, 0.0329, 0.0862, 0.1180, 0.0654,
                           0.0091, 0.0110, 0.0441, 0.0880, 0.0169],
                          [0.0141, 0.0229, 0.0231, 0.0162, 0.0160, 0.0380, 0.0375, 0.0216,
                           0.0278, 0.1099, 0.1371, 0.0554, 0.0422, 0.0787, 0.0965, 0.0658,
                           0.0183, 0.0207, 0.0509, 0.0798, 0.0274],
                          [0.0177, 0.0266, 0.0269, 0.0200, 0.0198, 0.0407, 0.0402, 0.0254,
                           0.0314, 0.0990, 0.1190, 0.0558, 0.0444, 0.0748, 0.0887, 0.0644,
                           0.0221, 0.0245, 0.0520, 0.0757, 0.0309],
                          [0.0103, 0.0185, 0.0188, 0.0122, 0.0120, 0.0342, 0.0337, 0.0173,
                           0.0235, 0.1237, 0.1616, 0.0541, 0.0388, 0.0826, 0.1057, 0.0665,
                           0.0141, 0.0164, 0.0488, 0.0840, 0.0230],
                          [0.0085, 0.0163, 0.0165, 0.0103, 0.0101, 0.0319, 0.0314, 0.0151,
                           0.0211, 0.1315, 0.1765, 0.0528, 0.0367, 0.0843, 0.1106, 0.0664,
                           0.0121, 0.0142, 0.0472, 0.0859, 0.0206],
                          [0.0121, 0.0207, 0.0210, 0.0142, 0.0140, 0.0362, 0.0356, 0.0194,
                           0.0257, 0.1167, 0.1489, 0.0549, 0.0406, 0.0808, 0.1011, 0.0663,
                           0.0162, 0.0185, 0.0500, 0.0820, 0.0252],
                          [0.0333, 0.0392, 0.0394, 0.0349, 0.0348, 0.0465, 0.0463, 0.0385,
                           0.0419, 0.0665, 0.0717, 0.0528, 0.0482, 0.0595, 0.0637, 0.0560,
                           0.0364, 0.0379, 0.0513, 0.0597, 0.0416],
                          [0.0306, 0.0373, 0.0375, 0.0324, 0.0323, 0.0459, 0.0457, 0.0365,
                           0.0404, 0.0710, 0.0778, 0.0536, 0.0480, 0.0619, 0.0673, 0.0575,
                           0.0341, 0.0358, 0.0518, 0.0623, 0.0402],
                          [0.0151, 0.0240, 0.0243, 0.0173, 0.0171, 0.0388, 0.0384, 0.0227,
                           0.0289, 0.1065, 0.1314, 0.0556, 0.0429, 0.0776, 0.0942, 0.0654,
                           0.0194, 0.0218, 0.0513, 0.0786, 0.0284],
                          [0.0102, 0.0184, 0.0186, 0.0121, 0.0119, 0.0340, 0.0335, 0.0171,
                           0.0233, 0.1243, 0.1626, 0.0540, 0.0387, 0.0827, 0.1061, 0.0665,
                           0.0140, 0.0162, 0.0487, 0.0842, 0.0228],
                          [0.0249, 0.0330, 0.0332, 0.0271, 0.0269, 0.0442, 0.0439, 0.0320,
                           0.0370, 0.0817, 0.0928, 0.0550, 0.0470, 0.0674, 0.0758, 0.0608,
                           0.0290, 0.0311, 0.0524, 0.0679, 0.0366]]])

    v = torch.tensor([[[0.2436],
                       [0.5590],
                       [0.5573],
                       [0.2458],
                       [0.2539],
                       [0.8085],
                       [0.8692],
                       [0.5824],
                       [0.7743],
                       [1.7807],
                       [1.9131],
                       [1.1855],
                       [1.2084],
                       [1.8202],
                       [1.7468],
                       [1.2375],
                       [0.3530],
                       [0.4253],
                       [1.3957],
                       [1.4435],
                       [0.7049]]])

    # print(torch.sum(dist))
    # att = torch.bmm(dist, v)
    # print(att)
    # print(torch.sum(dist,dim=1))
    # print(att/v)
    # print(1.024/v)
    # a = torch.tensor([[[1, 2, 2, 4]]]).float()
    # a = torch.transpose(a, 1, 2)
    # q = a
    # k = a
    # v = a
    # k = torch.transpose(k, 1, 2)
    #
    # dist = torch.bmm(q, k)  # batch, n, n
    # dist = torch.softmax(dist, dim=-1)  # batch, n, n
    # # print(dist, torch.sum(dist))
    #
    # # v = torch.transpose(v, 1, 2)
    # att = torch.bmm(dist, v)
    # print(1)
    # import sys
    #
    # sys.path.append('..')
    # import torch
    # from itertools import chain
    # import os
    #
    # from src.multimodal_to_rgb_kd_dataloader import multimodal_to_rgb_kd_dataloader
    # from src.surf_baseline_multi_dataloader import surf_baseline_multi_dataloader
    # from models.surf_patch_spp import SURF_Patch_SPP
    # from models.resnet_se_spp import resnet18_se_patch_spp
    # from loss.kd import *
    # from lib.model_develop import train_knowledge_distill_patch
    # from configuration.config_patch_kd_spp import args
    # import torch.nn as nn
    # from thop import profile
    #
    # args.modal = args.student_modal  # 用于获取指定的模态训练学生模型
    #
    # student_model = nn.Linear(512,1000,bias=True)
    #
    #
    # input = torch.rand((1, 512))
    # macs, params = profile(student_model, inputs=(input,))
    # print(macs, params)
