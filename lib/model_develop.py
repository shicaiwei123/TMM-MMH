'''模型训练相关的函数'''

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
import time
import csv
import os
from torchtoolbox.tools import mixup_criterion, mixup_data
import time

import os
from sklearn.manifold import TSNE
import torch.nn as nn
import torch.nn.functional as F

from lib.model_develop_utils import GradualWarmupScheduler, calc_accuracy
from loss.mmd_loss import MMD_loss


def calc_accuracy_multi_advisor(model, loader, args, verbose=False, hter=False):
    """
    :param model: model network
    :param loader: torch.utils.data.DataLoader
    :param verbose: show progress bar, bool
    :return accuracy, float
    """
    mode_saved = model.training
    model.train(False)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    outputs_full = []
    labels_full = []
    for sample_batch in tqdm(iter(loader), desc="Full forward pass", total=len(loader), disable=not verbose):
        img_rgb, ir_target, depth_target, binary_target = sample_batch['image_rgb'], sample_batch['image_ir'], \
                                                          sample_batch['image_depth'], sample_batch['binary_label']

        if torch.cuda.is_available():
            img_rgb, ir_target, depth_target, binary_target = img_rgb.cuda(), ir_target.cuda(), depth_target.cuda(), binary_target.cuda()

        with torch.no_grad():
            if args.method == 'deeppix':
                ir_out, depth_out, outputs_batch = model(img_rgb)
            elif args.method == 'pyramid':
                if args.origin_deeppix:
                    x, x, x, x, outputs_batch = model(img_rgb)
                else:
                    x, x, x, x, outputs_batch = model(img_rgb)
            else:
                print("test error")
        outputs_full.append(outputs_batch)
        labels_full.append(binary_target)
    model.train(mode_saved)
    outputs_full = torch.cat(outputs_full, dim=0)
    labels_full = torch.cat(labels_full, dim=0)
    _, labels_predicted = torch.max(outputs_full.data, dim=1)
    accuracy = torch.sum(labels_full == labels_predicted).item() / float(len(labels_full))
    accuracy = float("%.6f" % accuracy)

    if hter:
        predict_arr = np.array(labels_predicted.cpu())
        label_arr = np.array(labels_full.cpu())

        living_wrong = 0  # living -- spoofing
        living_right = 0
        spoofing_wrong = 0  # spoofing ---living
        spoofing_right = 0

        for i in range(len(predict_arr)):
            if predict_arr[i] == label_arr[i]:
                if label_arr[i] == 1:
                    living_right += 1
                else:
                    spoofing_right += 1
            else:
                # 错误
                if label_arr[i] == 1:
                    living_wrong += 1
                else:
                    spoofing_wrong += 1

        FRR = living_wrong / (living_wrong + living_right)
        APCER = living_wrong / (spoofing_right + living_wrong)
        NPCER = spoofing_wrong / (spoofing_wrong + living_right)
        FAR = spoofing_wrong / (spoofing_wrong + spoofing_right)
        HTER = (FAR + FRR) / 2

        FAR = float("%.6f" % FAR)
        FRR = float("%.6f" % FRR)
        HTER = float("%.6f" % HTER)
        accuracy = float("%.6f" % accuracy)

        return [accuracy, FAR, FRR, HTER, APCER, NPCER]
    else:
        return [accuracy]


def calc_accuracy_multi(model, loader, verbose=False, hter=False):
    """
    :param model: model network
    :param loader: torch.utils.data.DataLoader
    :param verbose: show progress bar, bool
    :return accuracy, float
    """
    mode_saved = model.training
    model.train(False)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    outputs_full = []
    labels_full = []

    for batch_sample in tqdm(iter(loader), desc="Full forward pass", total=len(loader), disable=not verbose):

        img_rgb, img_ir, img_depth, target = batch_sample['image_x'], batch_sample['image_ir'], \
                                             batch_sample['image_depth'], batch_sample[
                                                 'binary_label']

        if torch.cuda.is_available():
            img_rgb = img_rgb.cuda()
            img_ir = img_ir.cuda()
            img_depth = img_depth.cuda()
            target = target.cuda()

        with torch.no_grad():
            outputs_batch = model(img_rgb, img_ir, img_depth)
            if isinstance(outputs_batch, tuple):
                outputs_batch = outputs_batch[0]
            # print(outputs_batch)
        outputs_full.append(outputs_batch)
        labels_full.append(target)

    model.train(mode_saved)
    outputs_full = torch.cat(outputs_full, dim=0)
    labels_full = torch.cat(labels_full, dim=0)
    _, labels_predicted = torch.max(outputs_full.data, dim=1)
    accuracy = torch.sum(labels_full == labels_predicted).item() / float(len(labels_full))
    # print((labels_full - labels_predicted))
    accuracy = float("%.6f" % accuracy)

    if hter:
        predict_arr = np.array(labels_predicted.cpu())
        label_arr = np.array(labels_full.cpu())

        living_wrong = 0  # living -- spoofing
        living_right = 0
        spoofing_wrong = 0  # spoofing ---living
        spoofing_right = 0

        for i in range(len(predict_arr)):
            if predict_arr[i] == label_arr[i]:
                if label_arr[i] == 1:
                    living_right += 1
                else:
                    spoofing_right += 1
            else:
                # 错误
                if label_arr[i] == 1:
                    living_wrong += 1
                else:
                    spoofing_wrong += 1

        try:

            FRR = living_wrong / (living_wrong + living_right)
            APCER = living_wrong / (spoofing_right + living_wrong)
            NPCER = spoofing_wrong / (spoofing_wrong + living_right)
            FAR = spoofing_wrong / (spoofing_wrong + spoofing_right)
            HTER = (FAR + FRR) / 2

            FAR = float("%.6f" % FAR)
            FRR = float("%.6f" % FRR)
            HTER = float("%.6f" % HTER)
            accuracy = float("%.6f" % accuracy)
        except Exception as e:
            print(living_right, living_wrong, spoofing_right, spoofing_wrong)
            return [accuracy, 0, 0, 0, 0, 0]

        print(living_right, living_wrong, spoofing_right, spoofing_wrong)
        return [accuracy, FAR, FRR, HTER, APCER, NPCER]
    else:
        return [accuracy]


def calc_accuracy_ensemble(model, loader, verbose=False, hter=False):
    """
    :param model: model network
    :param loader: torch.utils.data.DataLoader
    :param verbose: show progress bar, bool
    :return accuracy, float
    """
    mode_saved = model.training
    model.train(False)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    outputs_full = []
    labels_full = []

    for batch_sample in tqdm(iter(loader), desc="Full forward pass", total=len(loader), disable=not verbose):

        img_rgb, img_ir, img_depth, target = batch_sample['image_x'], batch_sample['image_ir'], \
                                             batch_sample['image_depth'], batch_sample[
                                                 'binary_label']

        if torch.cuda.is_available():
            img_rgb = img_rgb.cuda()
            img_ir = img_ir.cuda()
            img_depth = img_depth.cuda()
            target = target.cuda()

        with torch.no_grad():

            outputs_batch = model(img_rgb)

            # 如果有多个返回值只取第一个
            if isinstance(outputs_batch, tuple):
                outputs_batch = outputs_batch[0]
            # print(outputs_batch)
        outputs_full.append(outputs_batch)
        labels_full.append(target)

    model.train(mode_saved)
    outputs_full = torch.cat(outputs_full, dim=0)
    labels_full = torch.cat(labels_full, dim=0)
    _, labels_predicted = torch.max(outputs_full.data, dim=1)
    accuracy = torch.sum(labels_full == labels_predicted).item() / float(len(labels_full))
    # print((labels_full - labels_predicted))
    accuracy = float("%.6f" % accuracy)

    if hter:
        predict_arr = np.array(labels_predicted.cpu())
        label_arr = np.array(labels_full.cpu())

        living_wrong = 0  # living -- spoofing
        living_right = 0
        spoofing_wrong = 0  # spoofing ---living
        spoofing_right = 0

        for i in range(len(predict_arr)):
            if predict_arr[i] == label_arr[i]:
                if label_arr[i] == 1:
                    living_right += 1
                else:
                    spoofing_right += 1
            else:
                # 错误
                if label_arr[i] == 1:
                    living_wrong += 1
                else:
                    spoofing_wrong += 1

        try:

            FRR = living_wrong / (living_wrong + living_right)
            APCER = living_wrong / (spoofing_right + living_wrong)
            NPCER = spoofing_wrong / (spoofing_wrong + living_right)
            ACER = (APCER + NPCER) / 2
            FAR = spoofing_wrong / (spoofing_wrong + spoofing_right)
            HTER = (FAR + FRR) / 2

            FAR = float("%.6f" % FAR)
            FRR = float("%.6f" % FRR)
            HTER = float("%.6f" % HTER)
            APCER = float("%.6f" % APCER)
            NPCER = float("%.6f" % NPCER)
            ACER = float("%.6f" % ACER)
            accuracy = float("%.6f" % accuracy)
        except Exception as e:
            print(e)
            print(living_right, living_wrong, spoofing_right, spoofing_wrong)
            return [accuracy, 1, 1, 1, 1, 1, 1]

        print(living_right, living_wrong, spoofing_right, spoofing_wrong)
        return [accuracy, FAR, FRR, HTER, APCER, NPCER, ACER]
    else:
        return [accuracy]


def calc_accuracy_kd(model, loader, args, verbose=False, hter=False):
    """
    :param model: model network
    :param loader: torch.utils.data.DataLoader
    :param verbose: show progress bar, bool
    :return accuracy, float
    """
    mode_saved = model.training
    model.train(False)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    outputs_full = []
    labels_full = []

    if args.student_data == 'multi_rgb' or args.student_data == 'multi_depth' or args.student_data == 'multi_ir':
        for batch_sample in tqdm(iter(loader), desc="Full forward pass", total=len(loader), disable=not verbose):

            img_rgb, img_ir, img_depth, target = batch_sample['image_x'], batch_sample['image_ir'], \
                                                 batch_sample['image_depth'], batch_sample[
                                                     'binary_label']

            if torch.cuda.is_available():
                img_rgb = img_rgb.cuda()
                img_ir = img_ir.cuda()
                img_depth = img_depth.cuda()
                target = target.cuda()

            if args.student_data == 'multi_rgb':
                test_data = img_rgb
            elif args.student_data == 'multi_depth':
                test_data = img_depth
            elif args.student_data == 'multi_ir':
                test_data = img_ir
            else:
                test_data = img_rgb
                print('test_error')
            with torch.no_grad():
                outputs_batch = model(test_data)

                # 如果有多个返回值只取第一个
                if isinstance(outputs_batch, tuple):
                    outputs_batch = outputs_batch[0]
            outputs_full.append(outputs_batch)
            labels_full.append(target)
    else:
        for (batch_sample, single_sample, label) in tqdm(iter(loader), desc="Full forward pass", total=len(loader),
                                                         disable=not verbose):

            if torch.cuda.is_available():
                single_sample = single_sample.cuda()
                label = label.cuda()

            with torch.no_grad():
                outputs_batch = model(single_sample)

                # 如果有多个返回值只取第一个
                if isinstance(outputs_batch, tuple):
                    outputs_batch = outputs_batch[0]

                # print(outputs_batch)
            outputs_full.append(outputs_batch)
            labels_full.append(label)

    model.train(mode_saved)
    outputs_full = torch.cat(outputs_full, dim=0)
    labels_full = torch.cat(labels_full, dim=0)
    _, labels_predicted = torch.max(outputs_full.data, dim=1)
    accuracy = torch.sum(labels_full == labels_predicted).item() / float(len(labels_full))
    # print((labels_full - labels_predicted))
    accuracy = float("%.6f" % accuracy)

    if hter:
        predict_arr = np.array(labels_predicted.cpu())
        label_arr = np.array(labels_full.cpu())

        living_wrong = 0  # living -- spoofing
        living_right = 0
        spoofing_wrong = 0  # spoofing ---living
        spoofing_right = 0

        for i in range(len(predict_arr)):
            if predict_arr[i] == label_arr[i]:
                if label_arr[i] == 1:
                    living_right += 1
                else:
                    spoofing_right += 1
            else:
                # 错误
                if label_arr[i] == 1:
                    living_wrong += 1
                else:
                    spoofing_wrong += 1

        try:

            FRR = living_wrong / (living_wrong + living_right)
            APCER = living_wrong / (spoofing_right + living_wrong)
            NPCER = spoofing_wrong / (spoofing_wrong + living_right)
            FAR = spoofing_wrong / (spoofing_wrong + spoofing_right)
            HTER = (FAR + FRR) / 2

            FAR = float("%.6f" % FAR)
            FRR = float("%.6f" % FRR)
            HTER = float("%.6f" % HTER)
            accuracy = float("%.6f" % accuracy)
        except Exception as e:
            print(living_right, living_wrong, spoofing_right, spoofing_wrong)
            return [accuracy, 0, 0, 0, 0, 0]

        print(living_right, living_wrong, spoofing_right, spoofing_wrong)
        return [accuracy, FAR, FRR, HTER, APCER, NPCER]
    else:
        return [accuracy]


def calc_accuracy_kd_admd(model, loader, args, verbose=False, hter=False):
    """
    :param model: model network
    :param loader: torch.utils.data.DataLoader
    :param verbose: show progress bar, bool
    :return accuracy, float
    """
    mode_saved = model.training
    model.train(False)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    outputs_full = []
    labels_full = []

    for (img_rgb, img_ir, img_depth, target) in tqdm(iter(loader), desc="Full forward pass", total=len(loader),
                                                     disable=not verbose):

        if torch.cuda.is_available():
            img_rgb = img_rgb.cuda()
            img_ir = img_ir.cuda()
            img_depth = img_depth.cuda()
            target = target.cuda()

        with torch.no_grad():
            outputs_batch = model(img_rgb)

            # 如果有多个返回值只取第一个
            if isinstance(outputs_batch, tuple):
                outputs_batch = outputs_batch[0]

            # print(outputs_batch)
        outputs_full.append(outputs_batch)
        labels_full.append(target)

    model.train(mode_saved)
    outputs_full = torch.cat(outputs_full, dim=0)
    labels_full = torch.cat(labels_full, dim=0)
    _, labels_predicted = torch.max(outputs_full.data, dim=1)
    accuracy = torch.sum(labels_full == labels_predicted).item() / float(len(labels_full))
    # print((labels_full - labels_predicted))
    accuracy = float("%.6f" % accuracy)

    if hter:
        predict_arr = np.array(labels_predicted.cpu())
        label_arr = np.array(labels_full.cpu())

        living_wrong = 0  # living -- spoofing
        living_right = 0
        spoofing_wrong = 0  # spoofing ---living
        spoofing_right = 0

        for i in range(len(predict_arr)):
            if predict_arr[i] == label_arr[i]:
                if label_arr[i] == 1:
                    living_right += 1
                else:
                    spoofing_right += 1
            else:
                # 错误
                if label_arr[i] == 1:
                    living_wrong += 1
                else:
                    spoofing_wrong += 1

        try:

            FRR = living_wrong / (living_wrong + living_right)
            APCER = living_wrong / (spoofing_right + living_wrong)
            NPCER = spoofing_wrong / (spoofing_wrong + living_right)
            FAR = spoofing_wrong / (spoofing_wrong + spoofing_right)
            HTER = (FAR + FRR) / 2

            FAR = float("%.6f" % FAR)
            FRR = float("%.6f" % FRR)
            HTER = float("%.6f" % HTER)
            accuracy = float("%.6f" % accuracy)
        except Exception as e:
            print(living_right, living_wrong, spoofing_right, spoofing_wrong)
            return [accuracy, 0, 0, 0, 0, 0]

        print(living_right, living_wrong, spoofing_right, spoofing_wrong)
        return [accuracy, FAR, FRR, HTER, APCER, NPCER]
    else:
        return [accuracy]


def calc_accuracy_kd_patch(model, loader, args, verbose=False, hter=False):
    """
    :param model: model network
    :param loader: torch.utils.data.DataLoader
    :param verbose: show progress bar, bool
    :return accuracy, float
    """
    mode_saved = model.training
    model.train(False)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    outputs_full = []
    labels_full = []

    if args.student_data == 'multi_rgb':
        for batch_sample in tqdm(iter(loader), desc="Full forward pass", total=len(loader), disable=not verbose):

            img_rgb, img_ir, img_depth, target = batch_sample['image_x'], batch_sample['image_ir'], \
                                                 batch_sample['image_depth'], batch_sample[
                                                     'binary_label']

            if torch.cuda.is_available():
                img_rgb = img_rgb.cuda()
                img_ir = img_ir.cuda()
                img_depth = img_depth.cuda()
                target = target.cuda()

            with torch.no_grad():
                outputs_batch = model(img_rgb)[0]

                # print(outputs_batch)
            outputs_full.append(outputs_batch)
            labels_full.append(target)
    else:
        for (batch_sample, single_sample, label) in tqdm(iter(loader), desc="Full forward pass", total=len(loader),
                                                         disable=not verbose):

            if torch.cuda.is_available():
                single_sample = single_sample.cuda()
                label = label.cuda()

            with torch.no_grad():
                outputs_batch = model(single_sample)
                # print(outputs_batch)
            outputs_full.append(outputs_batch)
            labels_full.append(label)

    model.train(mode_saved)
    outputs_full = torch.cat(outputs_full, dim=0)
    labels_full = torch.cat(labels_full, dim=0)
    _, labels_predicted = torch.max(outputs_full.data, dim=1)
    accuracy = torch.sum(labels_full == labels_predicted).item() / float(len(labels_full))
    # print((labels_full - labels_predicted))
    accuracy = float("%.6f" % accuracy)

    if hter:
        predict_arr = np.array(labels_predicted.cpu())
        label_arr = np.array(labels_full.cpu())

        living_wrong = 0  # living -- spoofing
        living_right = 0
        spoofing_wrong = 0  # spoofing ---living
        spoofing_right = 0

        for i in range(len(predict_arr)):
            if predict_arr[i] == label_arr[i]:
                if label_arr[i] == 1:
                    living_right += 1
                else:
                    spoofing_right += 1
            else:
                # 错误
                if label_arr[i] == 1:
                    living_wrong += 1
                else:
                    spoofing_wrong += 1

        try:

            FRR = living_wrong / (living_wrong + living_right)
            APCER = living_wrong / (spoofing_right + living_wrong)
            NPCER = spoofing_wrong / (spoofing_wrong + living_right)
            ACER = (APCER + NPCER) / 2
            FAR = spoofing_wrong / (spoofing_wrong + spoofing_right)
            HTER = (FAR + FRR) / 2

            FAR = float("%.6f" % FAR)
            FRR = float("%.6f" % FRR)
            HTER = float("%.6f" % HTER)
            accuracy = float("%.6f" % accuracy)
        except Exception as e:
            print(living_right, living_wrong, spoofing_right, spoofing_wrong)
            return [accuracy, 1, 1, 1, 1, 1, 1]

        print(living_right, living_wrong, spoofing_right, spoofing_wrong)
        return [accuracy, FAR, FRR, HTER, APCER, NPCER, ACER]
    else:
        return [accuracy]


def calc_accuracy_kd_patch_feature(model, loader, args, verbose=False, hter=False):
    """
    :param model: model network
    :param loader: torch.utils.data.DataLoader
    :param verbose: show progress bar, bool
    :return accuracy, float
    """
    mode_saved = model.training
    model.train(False)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    outputs_full = []
    labels_full = []

    if args.student_data == 'multi_rgb':
        for batch_sample in tqdm(iter(loader), desc="Full forward pass", total=len(loader), disable=not verbose):

            img_rgb, img_ir, img_depth, target = batch_sample['image_x'], batch_sample['image_ir'], \
                                                 batch_sample['image_depth'], batch_sample[
                                                     'binary_label']

            if torch.cuda.is_available():
                img_rgb = img_rgb.cuda()
                img_ir = img_ir.cuda()
                img_depth = img_depth.cuda()
                target = target.cuda()

            with torch.no_grad():
                outputs_batch, x, x, x = model(img_rgb)
                # print(outputs_batch)
            outputs_full.append(outputs_batch)
            labels_full.append(target)
    else:
        for (batch_sample, single_sample, label) in tqdm(iter(loader), desc="Full forward pass", total=len(loader),
                                                         disable=not verbose):

            if torch.cuda.is_available():
                single_sample = single_sample.cuda()
                label = label.cuda()

            with torch.no_grad():
                outputs_batch = model(single_sample)
                # print(outputs_batch)
            outputs_full.append(outputs_batch)
            labels_full.append(label)

    model.train(mode_saved)
    outputs_full = torch.cat(outputs_full, dim=0)
    labels_full = torch.cat(labels_full, dim=0)
    _, labels_predicted = torch.max(outputs_full.data, dim=1)
    accuracy = torch.sum(labels_full == labels_predicted).item() / float(len(labels_full))
    # print((labels_full - labels_predicted))
    accuracy = float("%.6f" % accuracy)

    # torch.unsqueeze()
    margin_hook = torch.cat((outputs_full, labels_full.float().unsqueeze(1)), dim=1)

    if hter:
        predict_arr = np.array(labels_predicted.cpu())
        label_arr = np.array(labels_full.cpu())

        living_wrong = 0  # living -- spoofing
        living_right = 0
        spoofing_wrong = 0  # spoofing ---living
        spoofing_right = 0

        for i in range(len(predict_arr)):
            if predict_arr[i] == label_arr[i]:
                if label_arr[i] == 1:
                    living_right += 1
                else:
                    spoofing_right += 1
            else:
                # 错误
                if label_arr[i] == 1:
                    living_wrong += 1
                else:
                    spoofing_wrong += 1

        try:

            FRR = living_wrong / (living_wrong + living_right)
            APCER = living_wrong / (spoofing_right + living_wrong)
            NPCER = spoofing_wrong / (spoofing_wrong + living_right)
            FAR = spoofing_wrong / (spoofing_wrong + spoofing_right)
            HTER = (FAR + FRR) / 2
            ACER = (APCER + NPCER) / 2

            FAR = float("%.6f" % FAR)
            FRR = float("%.6f" % FRR)
            HTER = float("%.6f" % HTER)
            accuracy = float("%.6f" % accuracy)
        except Exception as e:
            print(living_right, living_wrong, spoofing_right, spoofing_wrong)
            return [accuracy, 0, 0, 0, 0, 0, 0], margin_hook

        print(living_right, living_wrong, spoofing_right, spoofing_wrong)
        return [accuracy, FAR, FRR, HTER, APCER, NPCER, ACER], margin_hook
    else:
        return [accuracy], margin_hook


def calc_accuracy_kd_sift_patch(model, loader, args, verbose=False, hter=False):
    """
    :param model: model network
    :param loader: torch.utils.data.DataLoader
    :param verbose: show progress bar, bool
    :return accuracy, float
    """
    mode_saved = model.training
    model.train(False)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    outputs_full = []
    labels_full = []

    for batch_sample in tqdm(iter(loader), desc="Full forward pass", total=len(loader), disable=not verbose):

        img_rgb, img_ir, img_depth, target, cluster_center = batch_sample['image_x'], batch_sample['image_ir'], \
                                                             batch_sample['image_depth'], batch_sample[
                                                                 'binary_label'], batch_sample['cluster_center']

        if torch.cuda.is_available():
            img_rgb = img_rgb.cuda()
            img_ir = img_ir.cuda()
            img_depth = img_depth.cuda()
            target = target.cuda()

        with torch.no_grad():
            outputs_batch, x = model(img_rgb, cluster_center)
            # print(outputs_batch)
        outputs_full.append(outputs_batch)
        labels_full.append(target)

    model.train(mode_saved)
    outputs_full = torch.cat(outputs_full, dim=0)
    labels_full = torch.cat(labels_full, dim=0)
    _, labels_predicted = torch.max(outputs_full.data, dim=1)
    accuracy = torch.sum(labels_full == labels_predicted).item() / float(len(labels_full))
    # print((labels_full - labels_predicted))
    accuracy = float("%.6f" % accuracy)

    if hter:
        predict_arr = np.array(labels_predicted.cpu())
        label_arr = np.array(labels_full.cpu())

        living_wrong = 0  # living -- spoofing
        living_right = 0
        spoofing_wrong = 0  # spoofing ---living
        spoofing_right = 0

        for i in range(len(predict_arr)):
            if predict_arr[i] == label_arr[i]:
                if label_arr[i] == 1:
                    living_right += 1
                else:
                    spoofing_right += 1
            else:
                # 错误
                if label_arr[i] == 1:
                    living_wrong += 1
                else:
                    spoofing_wrong += 1

        try:

            FRR = living_wrong / (living_wrong + living_right)
            APCER = living_wrong / (spoofing_right + living_wrong)
            NPCER = spoofing_wrong / (spoofing_wrong + living_right)
            FAR = spoofing_wrong / (spoofing_wrong + spoofing_right)
            HTER = (FAR + FRR) / 2

            FAR = float("%.6f" % FAR)
            FRR = float("%.6f" % FRR)
            HTER = float("%.6f" % HTER)
            accuracy = float("%.6f" % accuracy)
        except Exception as e:
            print(living_right, living_wrong, spoofing_right, spoofing_wrong)
            return [accuracy, 0, 0, 0, 0, 0]

        print(living_right, living_wrong, spoofing_right, spoofing_wrong)
        return [accuracy, FAR, FRR, HTER, APCER, NPCER]
    else:
        return [accuracy]


def calc_accuracy_pixel(model, loader, verbose=False, hter=False):
    """
    :param model: model network
    :param loader: torch.utils.data.DataLoader
    :param verbose: show progress bar, bool
    :return accuracy, float
    """
    mode_saved = model.training
    measure = nn.MSELoss()
    measure_loss = 0
    model.train(False)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    for inputs, labels in tqdm(iter(loader), desc="Full forward pass", total=len(loader), disable=not verbose):
        if use_cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
        with torch.no_grad():
            outputs_batch = model(inputs)
        measure_loss += measure(outputs_batch, labels)
    model.train(mode_saved)

    return measure_loss / len(loader)


def train_multi_advsor(model, cost, optimizer, train_loader, test_loader, args):
    '''

    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''

    print(args)
    criterion_absolute_loss = nn.BCELoss()
    # Initialize and open timer
    start = time.time()

    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(args.train_epoch * 1 / 6),
                                                                              np.int(args.train_epoch * 2 / 6),
                                                                              np.int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    epoch = 0
    accuracy_best = 0
    log_list = []  # log need to save

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        for batch_idx, sample_batch in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            img_rgb, ir_target, depth_target, binary_target = sample_batch['image_rgb'], sample_batch['image_ir'], \
                                                              sample_batch['image_depth'], sample_batch['binary_label']

            ir_target = torch.unsqueeze(ir_target, 1)
            depth_target = torch.unsqueeze(depth_target, 1)

            batch_num += 1
            if torch.cuda.is_available():
                img_rgb, ir_target, depth_target, binary_target = img_rgb.cuda(), ir_target.cuda(), depth_target.cuda(), binary_target.cuda()

            optimizer.zero_grad()

            batch_size = img_rgb.shape[0]
            index = binary_target.cpu()
            index = torch.unsqueeze(index, 1)
            y_one_hot = torch.zeros(batch_size, args.class_num).scatter_(1, index, 1)
            if torch.cuda.is_available():
                y_one_hot = y_one_hot.cuda()

            if args.method == 'deeppix':
                ir_out, depth_out, binary_out = model(img_rgb)
                loss1 = cost(binary_out, y_one_hot)
                loss2 = criterion_absolute_loss(ir_out, ir_target.float())
                loss3 = criterion_absolute_loss(depth_out, depth_target.float())
                loss = (loss1 + loss2 + loss3) / 3
            elif args.method == 'pyramid':
                if args.origin_deeppix:
                    out_8x8, out_4x4, out_2x2, out_1x1, binary_out = model(img_rgb)
                    loss1 = cost(binary_out, y_one_hot)
                    depth_target_8x8 = F.adaptive_avg_pool2d(depth_target, (8, 8))
                    depth_target_4x4 = F.adaptive_avg_pool2d(depth_target, (4, 4))
                    depth_target_2x2 = F.adaptive_avg_pool2d(depth_target, (2, 2))
                    depth_target_1x1 = F.adaptive_avg_pool2d(depth_target, (1, 1))

                    ir_target_8x8 = F.adaptive_avg_pool2d(ir_target, (8, 8))
                    ir_target_4x4 = F.adaptive_avg_pool2d(ir_target, (4, 4))
                    ir_target_2x2 = F.adaptive_avg_pool2d(ir_target, (2, 2))
                    ir_target_1x1 = F.adaptive_avg_pool2d(ir_target, (1, 1))

                    loss2 = criterion_absolute_loss(out_8x8, depth_target_8x8.float())
                    loss3 = criterion_absolute_loss(out_4x4, depth_target_4x4.float())
                    loss4 = criterion_absolute_loss(out_2x2, depth_target_2x2.float())
                    loss5 = criterion_absolute_loss(out_1x1, depth_target_1x1.float())

                    loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5
                else:
                    out_depth_32x32, out_depth_16x16, out_ir_32x32, out_ir_16x16, binary_out = model(img_rgb)
                    loss1 = cost(binary_out, y_one_hot)

                    depth_target_32x32 = F.adaptive_avg_pool2d(depth_target, (32, 32))
                    depth_target_16x16 = F.adaptive_avg_pool2d(depth_target, (16, 16))

                    ir_target_32x32 = F.adaptive_avg_pool2d(ir_target, (32, 32))
                    ir_target_16x16 = F.adaptive_avg_pool2d(ir_target, (16, 16))

                    loss2 = criterion_absolute_loss(out_depth_32x32, depth_target_32x32.float())
                    loss3 = criterion_absolute_loss(out_ir_32x32, ir_target_32x32.float())
                    loss4 = criterion_absolute_loss(out_depth_16x16, depth_target_16x16.float())
                    loss5 = criterion_absolute_loss(out_ir_16x16, ir_target_16x16.float())
                    loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5

            else:
                print("loss error")
                loss = torch.tensor(0)

            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            # if batch_idx % log_interval == 0:  # 准备打印相关信息，args.log_interval是最开头设置的好了的参数
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx * len(data), len(train_loader.dataset),
            #                100. * batch_idx / len(train_loader), loss.item()))

        # testing
        result_test = calc_accuracy_multi_advisor(model, loader=test_loader, args=args)
        accuracy_test = result_test[0]
        if accuracy_test > accuracy_best and epoch > 12:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(model.state_dict(), save_path, )
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        log_list.append(torch.detach(loss1).cpu().numpy())
        log_list.append(torch.detach(loss2).cpu().numpy())
        log_list.append(torch.detach(loss3).cpu().numpy())

        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(train_loader),
                                                                                        accuracy_test, accuracy_best))
        train_loss = 0

        print(loss1.cpu(), loss2.cpu(), loss3.cpu())
        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)


def train_base_multi(model, cost, optimizer, train_loader, test_loader, args):
    '''
    适用于多模态分类的基础训练函数
    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''
    print(args)

    # Initialize and open timer
    start = time.time()

    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(args.train_epoch * 1 / 6),
                                                                              np.int(args.train_epoch * 2 / 6),
                                                                              np.int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    epoch = 0
    accuracy_best = 0
    log_list = []  # log need to save

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_rgb, img_ir, img_depth, target = batch_sample['image_x'], batch_sample['image_ir'], \
                                                 batch_sample['image_depth'], batch_sample[
                                                     'binary_label']
            if epoch == 0:
                continue
            if torch.cuda.is_available():
                img_rgb = img_rgb.cuda()
                img_ir = img_ir.cuda()
                img_depth = img_depth.cuda()
                target = target.cuda()

            optimizer.zero_grad()

            output = model(img_rgb, img_ir, img_depth)
            if isinstance(output, tuple):
                output = output[0]

            loss = cost(output, target)

            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        # testing
        result_test = calc_accuracy_multi(model, loader=test_loader, hter=True, verbose=True)
        accuracy_test = result_test[0]
        if accuracy_test > accuracy_best and epoch > 5:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(model.state_dict(), save_path)
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(train_loader),
                                                                                        accuracy_test, accuracy_best))
        train_loss = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)


def train_ensemble(model, cost, optimizer, train_loader, test_loader, args):
    '''
    适用于多模态分类的基础训练函数
    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''
    print(args)

    # Initialize and open timer
    start = time.time()

    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(args.train_epoch * 1 / 6),
                                                                              np.int(args.train_epoch * 2 / 6),
                                                                              np.int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    epoch = 0
    accuracy_best = 0
    acer_best = 1
    hter_best = 1
    log_list = []  # log need to save

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_rgb, img_ir, img_depth, target = batch_sample['image_x'], batch_sample['image_ir'], \
                                                 batch_sample['image_depth'], batch_sample[
                                                     'binary_label']
            if epoch == 0:
                continue
            if torch.cuda.is_available():
                img_rgb = img_rgb.cuda()
                img_ir = img_ir.cuda()
                img_depth = img_depth.cuda()
                target = target.cuda()

            optimizer.zero_grad()

            output = model(img_rgb)

            loss = cost(output, target)

            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        # testing
        print(model.fuse_weight_1.cpu())
        result_test = calc_accuracy_ensemble(model, loader=test_loader, hter=True)
        print(result_test)
        accuracy_test = result_test[0]
        hter_test = result_test[3]
        acer_test = result_test[-1]

        if acer_test < acer_best and epoch > 0:
            acer_best = acer_test
            save_path = os.path.join(args.model_root, args.name + '_acer_best_' + '.pth')
            torch.save(model.state_dict(), save_path, )

        if hter_test < hter_best and epoch > 0:
            hter_best = hter_test
            save_path = os.path.join(args.model_root, args.name + '_hter_best_' + '.pth')
            torch.save(model.state_dict(), save_path, )

        if acer_test < hter_test and acer_test < 0.65 and epoch % 2 == 0:
            save_path = os.path.join(args.model_root, args.name + '_' + str(epoch) + '_.pth')
            torch.save(model.state_dict(), save_path, )

        if accuracy_test > accuracy_best and epoch > 0:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(model.state_dict(), save_path, )
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(train_loader),
                                                                                        accuracy_test, accuracy_best))
        train_loss = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)


def train_pixel_supervise(model, cost, optimizer, train_loader, test_loader, args):
    '''

    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''

    print(args)

    # Initialize and open timer
    start = time.time()

    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    # Cosine learning rate decay
    if args.lr_decrease:
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    epoch = 0
    loss_best = 1e4
    log_list = []  # log need to save

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        for batch_idx, (data, target) in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1

            target = torch.unsqueeze(target, dim=1)

            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            if args.mixup:
                mixup_alpha = args.mixup_alpha
                inputs, labels_a, labels_b, lam = mixup_data(data, target, alpha=mixup_alpha)

            optimizer.zero_grad()

            output = model(data)

            if args.mixup:
                loss = mixup_criterion(cost, output, labels_a, labels_b, lam)
            else:
                loss = cost(output, target)

            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            # if batch_idx % log_interval == 0:  # 准备打印相关信息，args.log_interval是最开头设置的好了的参数
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx * len(data), len(train_loader.dataset),
            #                100. * batch_idx / len(train_loader), loss.item()))

        # testing
        result_test = calc_accuracy_pixel(model, loader=test_loader)
        test_loss = result_test
        if test_loss < loss_best:
            loss_best = train_loss / len(train_loader)
            save_path = args.model_root + args.name + '.pth'
            torch.save(model.state_dict(), save_path)
        log_list.append(test_loss)

        print(
            "Epoch {}, loss={:.5f}".format(epoch,
                                           train_loss / len(train_loader),
                                           ))
        train_loss = 0
        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)


def train_knowledge_distill(net_dict, cost_dict, optimizer, train_loader, test_loader, args):
    '''

    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''

    print(args)
    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    student_model = net_dict['snet']
    teacher_model = net_dict['tnet']

    criterionCls = cost_dict['criterionCls']
    criterionKD = cost_dict['criterionKD']

    if torch.cuda.is_available():
        mse_loss = nn.MSELoss().cuda()
    else:
        mse_loss = nn.MSELoss()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(args.train_epoch * 1 / 6),
                                                                              np.int(args.train_epoch * 2 / 6),
                                                                              np.int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    cls_loss_sum = 0
    kd_loss_sum = 0
    epoch = 0
    accuracy_best = 0
    log_list = []  # log need to save

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            student_model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        import datetime
        start = datetime.datetime.now()

        for batch_idx, (multi_sample, single_sample, label) in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            if epoch == 0:
                continue

            data_read_time = (datetime.datetime.now() - start)
            # print("data_read_time:", data_read_time.total_seconds())
            start = datetime.datetime.now()
            batch_num += 1

            img_rgb, img_ir, img_depth, target = multi_sample['image_x'], multi_sample['image_ir'], \
                                                 multi_sample['image_depth'], multi_sample[
                                                     'binary_label']

            if torch.cuda.is_available():
                img_rgb = img_rgb.cuda()
                img_ir = img_ir.cuda()
                img_depth = img_depth.cuda()
                target = target.cuda()
                single_sample = single_sample.cuda()
                label = label.cuda()

            optimizer.zero_grad()

            teacher_output, teacher_layer3, teacher_layer4 = teacher_model(img_rgb, img_ir, img_depth)

            if args.student_data == 'multi_rgb':
                student_out, student_layer3, student_layer4 = student_model(img_rgb)
            elif args.student_data == 'multi_depth':
                student_out, student_layer3, student_layer4 = student_model(img_depth)
            elif args.student_data == 'multi_ir':
                student_out, student_layer3, student_layer4 = student_model(img_ir)
            else:
                student_out, student_layer3, student_layer4 = student_model(single_sample)

            time_forward = datetime.datetime.now() - start
            # print("time_forward:", time_forward.total_seconds())

            # 蒸馏logits损失
            if args.kd_mode in ['logits', 'st']:
                kd_logits_loss = criterionKD(student_out, teacher_output.detach())
            else:
                kd_logits_loss = 0
                print("kd_Loss error")

                # 蒸馏feature
                # kd_feature_loss = mse_loss(torch.mean(teacher_layer3, dim=1), torch.mean(student_layer3, dim=1))
            kd_feature_loss = mse_loss(teacher_layer3, student_layer3)
            # 分类损失
            student_out = student_model.dropout(student_out)

            cls_loss = criterionCls(student_out, target)

            if args.logits_kd and args.feature_kd:
                kd_loss = kd_logits_loss + kd_feature_loss
            elif args.logits_kd:
                kd_loss = kd_logits_loss
            elif args.feature_kd:
                kd_loss = kd_feature_loss
            else:
                kd_loss = 0
            loss = cls_loss + kd_loss * args.lambda_kd

            train_loss += loss.item()
            cls_loss_sum += cls_loss.item()
            kd_loss_sum += kd_loss.item()
            loss.backward()
            optimizer.step()
            # if batch_idx % log_interval == 0:  # 准备打印相关信息，args.log_interval是最开头设置的好了的参数
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx * len(data), len(train_loader.dataset),
            #                100. * batch_idx / len(train_loader), loss.item()))

        # testing
        result_test = calc_accuracy_kd(model=student_model, args=args, loader=test_loader)
        accuracy_test = result_test[0]
        if accuracy_test > accuracy_best and epoch >= 15:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(student_model.state_dict(), save_path, )
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(cls_loss_sum / len(train_loader), kd_loss_sum / (len(train_loader)))

        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(
                                                                                            train_loader),
                                                                                        accuracy_test, accuracy_best))
        train_loss = 0
        cls_loss_sum = 0
        kd_loss_sum = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": student_model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)


def train_tgt(src_encoder, tgt_encoder, critic,
              src_data_loader, tgt_data_loader, args):
    """Train encoder for target domain."""
    print(args)
    acc_list = []
    import random
    from src.surf_baseline_single_dataloader import surf_baseline_single_dataloader
    from lib.model_develop_utils import calc_accuracy
    args.modal = 'rgb'
    val_data_loader = surf_baseline_single_dataloader(train=False, args=args)
    x_dropout = nn.Dropout(0.5)
    val_best = 0

    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    tgt_encoder.train()
    critic.train()

    # setup criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_tgt = optim.Adam(tgt_encoder.parameters(),
                               lr=args.d_learning_rate,
                               betas=(args.beta1, args.beta2))
    optimizer_critic = optim.Adam(critic.parameters(),
                                  lr=args.c_learning_rate,
                                  betas=(args.beta1, args.beta2))

    # optimizer_tgt = optim.RMSprop(tgt_encoder.parameters(),
    #                               lr=params.d_learning_rate,
    #                               # betas=(params.beta1, params.beta2)
    #                               )
    # optimizer_critic = optim.RMSprop(critic.parameters(),
    #                                  lr=params.c_learning_rate,
    #                                  # betas=(params.beta1, params.beta2)
    #                                  )
    #

    len_data_loader = min(len(src_data_loader), len(tgt_data_loader))
    print(len(src_data_loader), len(tgt_data_loader))
    ####################
    # 2. train network #
    ####################

    for epoch in range(args.train_epoch):
        # zip source and target data pair
        loss_class_sum = 0
        data_zip = enumerate(zip(src_data_loader, tgt_data_loader))
        for step, ((images_src, _), (images_tgt, label)) in data_zip:
            ###########################
            # 2.1 train discriminator #
            ###########################
            for _ in range(args.discriminator_iternum):
                # make images variable
                if torch.cuda.is_available():
                    images_src = images_src.cuda()
                    images_tgt = images_tgt.cuda()
                    label = label.cuda()

                # zero gradients for optimizer
                optimizer_critic.zero_grad()

                # extract and concat features
                x, feat_src = src_encoder(images_src)
                x_predict, feat_tgt = tgt_encoder(images_tgt)
                feat_concat = torch.cat((feat_src, feat_tgt), 0)

                # predict on discriminator
                pred_concat = critic(feat_concat.detach())

                # prepare real and fake label
                label_src = torch.ones(feat_src.size(0)).long()
                label_src[1:int(len(label_src) * 0.05)] = 0
                index = [i for i in range(label_src.shape[0])]
                random.shuffle(index)
                label_src = label_src[index]

                if torch.cuda.is_available():
                    label_src = label_src.cuda()

                label_tgt = torch.zeros(feat_tgt.size(0)).long()
                label_tgt[1:int(len(label_tgt) * 0.05)] = 1
                index = [i for i in range(label_tgt.shape[0])]
                random.shuffle(index)
                label_tgt = label_tgt[index]

                if torch.cuda.is_available():
                    label_tgt = label_tgt.cuda()

                label_concat = torch.cat((label_src, label_tgt), 0)

                # compute loss for critic
                loss_critic = criterion(pred_concat, label_concat)

                # add classify loss and critic loss
                loss_critic.backward()

                # optimize critic
                optimizer_critic.step()

            pred_cls = torch.squeeze(pred_concat.max(1)[1])
            acc = (pred_cls == label_concat).float().mean()

            ############################
            # 2.2 train target encoder # 训练目标域的特征提取器,使得提取的特征和源域尽可能接近
            ############################
            iter_time = 1
            for _ in range(args.generate_iternum):
                # zero gradients for optimizer
                optimizer_critic.zero_grad()
                optimizer_tgt.zero_grad()

                # extract and target features
                x_predict, feat_tgt = tgt_encoder(images_tgt)

                # predict on discriminator
                pred_tgt = critic(feat_tgt)

                # prepare fake labels
                label_tgt = torch.ones(feat_tgt.size(0)).long()
                if torch.cuda.is_available():
                    label_tgt = label_tgt.cuda()

                # compute loss for target encoder
                loss_tgt = criterion(pred_tgt, label_tgt)

                # compute loss for class
                x_predict = x_dropout(x_predict)
                loss_classify = criterion(x_predict, label)
                loss = loss_classify + loss_tgt
                loss_class_sum += loss_classify

                loss.backward()

            # optimize target encoder
            optimizer_tgt.step()

            #######################
            # 2.3 print step info #
            #######################
            if ((step + 1) % args.log_interval == 0):
                print("Epoch [{}/{}] Step [{}/{}]:"
                      "discriminator_loss={:.5f}  loss_tgt={:.5f},loss_class={:.5f} discriminator_acc={:.5f}  feature_dis={:.5f}  feature_norm2={:.5f}  val_best={:.5f}"
                      .format(epoch + 1,
                              args.train_epoch,
                              step + 1,
                              len_data_loader,
                              loss_critic.item(),
                              loss_tgt.item(),
                              loss_class_sum.item(),
                              acc.item(),
                              torch.sum(feat_tgt - feat_src).cpu().item(),
                              torch.norm(feat_tgt - feat_src).cpu().item(),
                              val_best
                              ))
            loss_class_sum = 0

        # test in each epoch
        result = calc_accuracy(model=tgt_encoder, loader=val_data_loader, verbose=True, hter=True)
        print(result)
        val_test = result[0]
        if val_test > val_best:
            val_best = val_test
            torch.save(tgt_encoder.state_dict(), os.path.join(
                args.model_root,
                args.method + "-" + args.teacher_data + "-" + args.student_data + "-val_best-" + "version-" + str(
                    args.version) + ".pth".format(
                    epoch + 1)))

        #############################
        # 2.4 save model parameters #
        #############################
        if ((epoch + 1) % args.save_interval == 0):
            torch.save(critic.state_dict(), os.path.join(
                args.model_root,
                args.method + "-" + args.teacher_data + "-" + args.student_data + "-critic-{}.pth".format(
                    epoch + 1)))
            torch.save(tgt_encoder.state_dict(), os.path.join(
                args.model_root,
                args.method + "-" + args.teacher_data + "-" + args.student_data + "-encoder-{}.pth".format(
                    epoch + 1)))

    torch.save(critic.state_dict(), os.path.join(
        args.model_root,
        args.method + "-" + args.teacher_data + "-" + args.student_data + "-critic-final.pth"))
    torch.save(tgt_encoder.state_dict(), os.path.join(
        args.model_root,
        args.method + "-" + args.teacher_data + "-" + args.student_data + "-encoder-final.pth"))
    return tgt_encoder


# def train_adversarial_knowledge_distill(net_dict, cost_dict, optimizer, train_loader, test_loader, args):
#     '''
#
#     :param model:
#     :param cost:
#     :param optimizer:
#     :param train_loader:
#     :param test_loader:
#     :param args:
#     :return:
#     '''
#
#     print(args)
#     if not os.path.exists(args.model_root):
#         os.makedirs(args.model_root)
#     if not os.path.exists(args.log_root):
#         os.makedirs(args.log_root)
#
#     models_dir = args.model_root + '/' + args.name + '.pt'
#     log_dir = args.log_root + '/' + args.name + '.csv'
#
#     # save args
#     with open(log_dir, 'a+', newline='') as f:
#         my_writer = csv.writer(f)
#         args_dict = vars(args)
#         for key, value in args_dict.items():
#             my_writer.writerow([key, value])
#         f.close()
#
#     student_model = net_dict['snet']
#     teacher_model = net_dict['tnet']
#
#     criterionCls = cost_dict['criterionCls']
#     criterionKD = cost_dict['criterionKD']
#
#     if torch.cuda.is_available():
#         mse_loss = nn.MSELoss().cuda()
#     else:
#         mse_loss = nn.MSELoss()
#
#     # 鉴别器
#     critic = Discriminator(args, input_dims=args.feat_dim)
#
#     #  learning rate decay
#     if args.lr_decrease == 'cos':
#         print("lrcos is using")
#         cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)
#
#         if args.lr_warmup:
#             scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
#                                                       after_scheduler=cos_scheduler)
#     elif args.lr_decrease == 'multi_step':
#         print("multi_step is using")
#         cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(args.train_epoch * 1 / 6),
#                                                                               np.int(args.train_epoch * 2 / 6),
#                                                                               np.int(args.train_epoch * 3 / 6)])
#         if args.lr_warmup:
#             scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
#                                                       after_scheduler=cos_scheduler)
#
#     # Training initialization
#     epoch_num = args.train_epoch
#     log_interval = args.log_interval
#     save_interval = args.save_interval
#     batch_num = 0
#     train_loss = 0
#     cls_loss_sum = 0
#     kd_loss_sum = 0
#     epoch = 0
#     accuracy_best = 0
#     log_list = []  # log need to save
#
#     if args.retrain:
#         if not os.path.exists(models_dir):
#             print("no trained model")
#         else:
#             state_read = torch.load(models_dir)
#             student_model.load_state_dict(state_read['model_state'])
#             optimizer.load_state_dict(state_read['optim_state'])
#             epoch = state_read['Epoch']
#             print("retaining")
#
#     # Train
#     while epoch < epoch_num:
#         import datetime
#         start = datetime.datetime.now()
#
#         for batch_idx, (teacher_data, student_data, label) in enumerate(
#                 tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):
#
#             if epoch == 0:
#                 continue
#
#             data_read_time = (datetime.datetime.now() - start)
#             # print("data_read_time:", data_read_time.total_seconds())
#             start = datetime.datetime.now()
#             batch_num += 1
#
#             if torch.cuda.is_available():
#                 teacher_data = teacher_data.cuda()
#                 student_data = student_data.cuda()
#                 label = label.cuda()
#
#             optimizer.zero_grad()
#
#             teacher_output, teacher_feature = teacher_model(teacher_data)
#
#             student_output, student_feature = student_model(student_data)
#
#             time_forward = datetime.datetime.now() - start
#             # print("time_forward:", time_forward.total_seconds())
#
#
#
#             # 蒸馏logits损失
#             if args.kd_mode in ['logits', 'st']:
#                 kd_logits_loss = criterionKD(student_out, teacher_output.detach())
#             else:
#                 kd_logits_loss = 0
#                 print("kd_Loss error")
#
#                 # 蒸馏feature
#                 # kd_feature_loss = mse_loss(torch.mean(teacher_layer3, dim=1), torch.mean(student_layer3, dim=1))
#             kd_feature_loss = mse_loss(teacher_layer3, student_layer3)
#             # 分类损失
#             student_out = student_model.dropout(student_out)
#
#             cls_loss = criterionCls(student_out, target)
#
#             if args.logits_kd and args.feature_kd:
#                 kd_loss = kd_logits_loss + kd_feature_loss
#             elif args.logits_kd:
#                 kd_loss = kd_logits_loss
#             elif args.feature_kd:
#                 kd_loss = kd_feature_loss
#             else:
#                 kd_loss = 0
#             loss = cls_loss + kd_loss * args.lambda_kd
#
#             train_loss += loss.item()
#             cls_loss_sum += cls_loss.item()
#             kd_loss_sum += kd_loss.item()
#             loss.backward()
#             optimizer.step()
#             # if batch_idx % log_interval == 0:  # 准备打印相关信息，args.log_interval是最开头设置的好了的参数
#             #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#             #         epoch, batch_idx * len(data), len(train_loader.dataset),
#             #                100. * batch_idx / len(train_loader), loss.item()))
#
#         # testing
#         result_test = calc_accuracy_kd(model=student_model, args=args, loader=test_loader)
#         accuracy_test = result_test[0]
#         if accuracy_test > accuracy_best and epoch >= 15:
#             accuracy_best = accuracy_test
#             save_path = os.path.join(args.model_root, args.name + '.pth')
#             torch.save(student_model.state_dict(), save_path, )
#         log_list.append(train_loss / len(train_loader))
#         log_list.append(accuracy_test)
#         log_list.append(accuracy_best)
#         print(cls_loss_sum / len(train_loader), kd_loss_sum / (len(train_loader)))
#
#         print(
#             "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
#                                                                                         train_loss / len(
#                                                                                             train_loader),
#                                                                                         accuracy_test, accuracy_best))
#         train_loss = 0
#         cls_loss_sum = 0
#         kd_loss_sum = 0
#
#         if args.lr_decrease:
#             if args.lr_warmup:
#                 scheduler_warmup.step(epoch=epoch)
#             else:
#                 cos_scheduler.step(epoch=epoch)
#         if epoch < 20:
#             print(epoch, optimizer.param_groups[0]['lr'])
#
#         # save model and para
#         if epoch % save_interval == 0:
#             train_state = {
#                 "Epoch": epoch,
#                 "model_state": student_model.state_dict(),
#                 "optim_state": optimizer.state_dict(),
#                 "args": args
#             }
#             models_dir = args.model_root + '/' + args.name + '.pt'
#             torch.save(train_state, models_dir)
#
#         #  save log
#         with open(log_dir, 'a+', newline='') as f:
#             # 训练结果
#             my_writer = csv.writer(f)
#             my_writer.writerow(log_list)
#             log_list = []
#         epoch = epoch + 1
#     train_duration_sec = int(time.time() - start)
#     print("training is end", train_duration_sec)


def train_self_distill(net_dict, cost_dict, optimizer, train_loader, test_loader, args):
    '''

      :param model:
      :param cost:
      :param optimizer:
      :param train_loader:
      :param test_loader:
      :param args:
      :return:
      '''

    print(args)
    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    student_model = net_dict['snet']
    teacher_model = net_dict['tnet']

    criterionCls = cost_dict['criterionCls']
    criterionKD = cost_dict['criterionKD']

    if torch.cuda.is_available():
        mmd_loss = MMD_loss().cuda()
    else:
        mmd_loss = MMD_loss()
    if torch.cuda.is_available():
        bce_loss = nn.BCELoss().cuda()
    else:
        bce_loss = nn.BCELoss()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(args.train_epoch * 1 / 6),
                                                                              np.int(args.train_epoch * 2 / 6),
                                                                              np.int(args.train_epoch * 5 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    cls_loss_sum = 0
    kd_loss_sum = 0
    epoch = 0
    accuracy_best = 0
    acer_best = 1
    hter_best = 1
    log_list = []  # log need to save

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            student_model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        import datetime
        start = datetime.datetime.now()

        for batch_idx, (multi_sample, single_sample, label) in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            if epoch == 0:
                continue

            data_read_time = (datetime.datetime.now() - start)
            # print("data_read_time:", data_read_time.total_seconds())
            start = datetime.datetime.now()
            batch_num += 1

            img_rgb, img_ir, img_depth, target = multi_sample['image_x'], multi_sample['image_ir'], \
                                                 multi_sample['image_depth'], multi_sample[
                                                     'binary_label']

            if torch.cuda.is_available():
                img_rgb = img_rgb.cuda()
                img_ir = img_ir.cuda()
                img_depth = img_depth.cuda()
                target = target.cuda()
                single_sample = single_sample.cuda()
                label = label.cuda()

            optimizer.zero_grad()

            teacher_whole_out, teacher_patch_out = teacher_model(img_rgb)

            if args.student_data == 'multi_rgb':
                if args.weight_patch:
                    student_whole_out, student_patch_out = student_model(img_rgb)
                else:
                    student_whole_out, student_patch_out = student_model(img_rgb)

            else:
                if args.weight_patch:
                    student_whole_out, student_patch_out = student_model(single_sample)
                else:
                    student_whole_out, student_patch_out = student_model(single_sample)
            time_forward = datetime.datetime.now() - start
            # print("time_forward:", time_forward.total_seconds())

            # 蒸馏损失
            if args.kd_mode in ['logits', 'st']:
                # patch_loss = mmd_loss(student_patch_out, teacher_patch_out.detach())
                # patch_loss = patch_loss.cuda()
                # whole_loss = criterionKD(student_whole_out, teacher_whole_out.detach())
                # whole_loss = whole_loss.cuda()
                # kd_loss = patch_loss + whole_loss
                # kd_loss = mmd_loss(student_patch_out, teacher_patch_out.detach())
                if args.weight_patch:
                    kd_loss = mmd_loss(student_patch_out, teacher_patch_out.detach())

                else:
                    kd_loss = mmd_loss(student_patch_out, teacher_patch_out.detach())
                kd_loss = kd_loss.cuda()
            else:
                kd_loss = 0
                print("kd_Loss error")

            # 分类损失
            if args.student_data == 'multi_rgb':
                cls_loss = criterionCls(student_whole_out, target)
            else:
                cls_loss = criterionCls(student_whole_out, label)

            cls_loss = cls_loss.cuda()

            loss = cls_loss + kd_loss * args.lambda_kd

            train_loss += loss.item()
            cls_loss_sum += cls_loss.item()
            kd_loss_sum += kd_loss.item()
            loss.backward()
            optimizer.step()
            # if batch_idx % log_interval == 0:  # 准备打印相关信息，args.log_interval是最开头设置的好了的参数
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx * len(data), len(train_loader.dataset),
            #                100. * batch_idx / len(train_loader), loss.item()))

        # testing
        result_test = calc_accuracy_kd_patch(model=student_model, args=args, loader=test_loader, hter=True)
        print(result_test)
        accuracy_test = result_test[0]
        hter_test = result_test[3]
        acer_test = result_test[-1]

        if acer_test < acer_best and epoch > 0:
            acer_best = acer_test
            save_path = os.path.join(args.model_root, args.name + '_acer_best_' + '.pth')
            torch.save(student_model.state_dict(), save_path, )

        if hter_test < hter_best and epoch > 0:
            hter_best = hter_test
            save_path = os.path.join(args.model_root, args.name + '_hter_best_' + '.pth')
            torch.save(student_model.state_dict(), save_path, )

        if accuracy_test > accuracy_best and epoch > 0:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(student_model.state_dict(), save_path, )
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(cls_loss_sum / len(train_loader), kd_loss_sum / (len(train_loader)))

        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(
                                                                                            train_loader),
                                                                                        accuracy_test, accuracy_best))
        train_loss = 0
        cls_loss_sum = 0
        kd_loss_sum = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": student_model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)


def train_knowledge_distill_patch(net_dict, cost_dict, optimizer, train_loader, test_loader, args):
    '''

    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''

    print(args)
    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    student_model = net_dict['snet']
    teacher_model = net_dict['tnet']

    criterionCls = cost_dict['criterionCls']
    criterionKD = cost_dict['criterionKD']

    from loss.kd.pkt import PKTCosSim

    if torch.cuda.is_available():
        mmd_loss = MMD_loss().cuda()
    else:
        mmd_loss = MMD_loss()
    if torch.cuda.is_available():
        bce_loss = nn.BCELoss().cuda()
    else:
        bce_loss = nn.BCELoss()

    if torch.cuda.is_available():
        pkd_loss = PKTCosSim().cuda()
    else:
        pkd_loss = PKTCosSim()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(args.train_epoch * 1 / 6),
                                                                              np.int(args.train_epoch * 2 / 6),
                                                                              np.int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    cls_loss_sum = 0
    kd_loss_sum = 0
    epoch = 0
    accuracy_best = 0
    acer_best = 1
    hter_best = 1
    log_list = []  # log need to save

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            student_model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        import datetime
        start = datetime.datetime.now()

        for batch_idx, (multi_sample, single_sample, label) in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            # if epoch == 0:
            #     continue

            data_read_time = (datetime.datetime.now() - start)
            # print("data_read_time:", data_read_time.total_seconds())
            start = datetime.datetime.now()
            batch_num += 1

            img_rgb, img_ir, img_depth, target = multi_sample['image_x'], multi_sample['image_ir'], \
                                                 multi_sample['image_depth'], multi_sample[
                                                     'binary_label']

            if torch.cuda.is_available():
                img_rgb = img_rgb.cuda()
                img_ir = img_ir.cuda()
                img_depth = img_depth.cuda()
                target = target.cuda()
                single_sample = single_sample.cuda()
                label = label.cuda()

            optimizer.zero_grad()

            teacher_whole_out, teacher_patch_out, teacher_patch_strength = teacher_model(img_rgb, img_ir, img_depth)

            if args.student_data == 'multi_rgb':
                student_whole_out, student_patch_out, student_patch_strength = student_model(img_rgb)
            else:
                student_whole_out, student_patch_out, student_patch_strength = student_model(single_sample)

            time_forward = datetime.datetime.now() - start
            # print("time_forward:", time_forward.total_seconds())

            # 蒸馏损失
            if args.kd_mode in ['logits', 'st', 'multi_st']:
                # patch_loss = mmd_loss(student_patch_out, teacher_patch_out.detach())
                # patch_loss = patch_loss.cuda()
                # whole_loss = criterionKD(student_whole_out, teacher_whole_out.detach())
                # whole_loss = whole_loss.cuda()
                # kd_loss = patch_loss + whole_loss
                # kd_loss = mmd_loss(student_patch_out, teacher_patch_out.detach())

                # multi kd/mmd
                # student_patch_out = torch.flatten(student_patch_out, start_dim=1,end_dim=2)
                # teacher_patch_out = torch.flatten(teacher_patch_out,start_dim=1,end_dim=2)
                # kd_loss = mmd_loss(student_patch_out, teacher_patch_out.detach())
                # print(teacher_patch_strength)

                # weight = torch.mean(student_patch_strength, dim=0)
                # weight = weight.cuda()
                # print(weight.shape)
                if args.weight_patch:
                    teacher_patch_strength = teacher_patch_strength.mean(dim=0)
                    kd_loss = criterionKD(student_patch_out, teacher_patch_out.detach(), weight=teacher_patch_strength)
                else:
                    kd_loss = criterionKD(student_patch_out, teacher_patch_out.detach(), weight=None)

                kd_loss = kd_loss.cuda()
                # print(kd_loss.is_cuda)
            else:
                kd_loss = 0
                print("kd_Loss error")

            # 分类损失
            if args.student_data == 'multi_rgb':
                cls_loss = criterionCls(student_whole_out, target)
            else:
                cls_loss = criterionCls(student_whole_out, label)

            cls_loss = cls_loss.cuda()

            loss = cls_loss + kd_loss * args.lambda_kd
            # print(loss.is_cuda)

            train_loss += loss.item()
            cls_loss_sum += cls_loss.item()
            kd_loss_sum += kd_loss.item()
            loss.backward()
            optimizer.step()
            # if batch_idx % log_interval == 0:  # 准备打印相关信息，args.log_interval是最开头设置的好了的参数
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx * len(data), len(train_loader.dataset),
            #                100. * batch_idx / len(train_loader), loss.item()))

        # testing
        result_test = calc_accuracy_kd_patch(model=student_model, args=args, loader=test_loader, hter=True)
        print(result_test)
        accuracy_test = result_test[0]
        hter_test = result_test[3]
        acer_test = result_test[-1]

        if acer_test < acer_best and epoch > 0:
            acer_best = acer_test
            save_path = os.path.join(args.model_root, args.name + '_acer_best_' + '.pth')
            torch.save(student_model.state_dict(), save_path, )

        if hter_test < hter_best and epoch > 0:
            hter_best = hter_test
            save_path = os.path.join(args.model_root, args.name + '_hter_best_' + '.pth')
            torch.save(student_model.state_dict(), save_path, )

        if accuracy_test > accuracy_best and epoch > 0:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(student_model.state_dict(), save_path, )
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(cls_loss_sum / len(train_loader), kd_loss_sum / (len(train_loader)))

        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(
                                                                                            train_loader),
                                                                                        accuracy_test, accuracy_best))
        train_loss = 0
        cls_loss_sum = 0
        kd_loss_sum = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": student_model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)


