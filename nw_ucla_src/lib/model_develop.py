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
from action_utils import util

import torchnet as tnt
import collections

from tensorboardX import SummaryWriter


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


def save(net, optimizer, epoch, iteration, model_dir, args):
    '''
    保存模型和args
    :param epoch:
    :param iteration:
    :return:
    '''
    from lib.processing_utils import makedir
    print('Saving state, iter:', iteration)
    state_dict = net.state_dict()
    optim_state = optimizer.state_dict()
    checkpoint = {'net': state_dict, 'optimizer': optim_state, 'iter': iteration}
    save_dir = args.model_root + '/' + args.method
    if not os.path.exists(save_dir):
        makedir(save_dir)
    torch.save(checkpoint, '%s/_ckpt_E_%d_I_%d.pth' % (save_dir, epoch, iteration))


def train_knowledge_distill_action(net_dict, cost_dict, optimizer, train_loader, test_loader, args):
    '''

    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''
    from lib.processing_utils import makedir
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

    writer = SummaryWriter('%s/tb.log' % (args.log_root + '/' + args.name))

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
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,
                                                                              150,
                                                                              300])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization

    cls_loss_sum = 0
    kd_loss_sum = 0
    epoch = 0
    average_acc = 0
    average_best = 0
    iteration = 0
    log_list = []  # log need to save
    loss_meters = collections.defaultdict(lambda: tnt.meter.MovingAverageValueMeter(20))

    total_iters = len(train_loader)
    print(total_iters)
    epoch = iteration // total_iters
    plot_every = int(0.1 * len(train_loader))

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            student_model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    while iteration <= args.max_iter:
        for batch in train_loader:

            rgb_batch = {"frames": batch["frames_rgb"], 'label': batch["label"]}
            depth_batch = {"frames": batch["frames_depth"], 'label': batch["label"]}

            if torch.cuda.is_available():
                rgb_batch = util.batch_cuda(rgb_batch)
                depth_batch = util.batch_cuda(depth_batch)

            pred_teacher, pred_spp_teacher, _, _ = teacher_model(rgb_batch, depth_batch)
            pred_student, pred_spp_student, student_loss_dict = student_model(rgb_batch)

            loss_dict = {k: v.mean() for k, v in student_loss_dict.items()}
            clc_loss = sum(loss_dict.values())

            kd_loss = criterionKD(pred_spp_student, pred_spp_teacher.detach())

            loss = clc_loss + kd_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, pred_idx = pred_student.max(1)
            # print(pred_idx)
            # print(pred_idx.is_cuda,batch['label'].is_cuda)
            pred_idx = pred_idx.cpu()
            correct = (pred_idx == batch['label']).float().sum()
            batch_acc = correct / pred_student.shape[0]
            average_acc += batch_acc.cpu().numpy()
            # print(batch_acc)
            loss_meters['bAcc'].add(batch_acc.item())

            for k, v in loss_dict.items():
                loss_meters[k].add(v.item())
            loss_meters['total_loss'].add(loss.item())

            # print(iteration)

            if iteration % args.print_every == 0:
                log_str = 'iter: %d (%d + %d/%d) | ' % (iteration, epoch, iteration % total_iters, total_iters)
                log_str += ' | '.join(['%s: %.3f' % (k, v.value()[0]) for k, v in loss_meters.items()])
                print(log_str, '|%f' % kd_loss.item())

            if iteration % args.plot_every == 0:
                for key in loss_meters:
                    writer.add_scalar('train/%s' % key, loss_meters[key].value()[0], int(100 * iteration / total_iters))

            iteration += 1

        epoch += 1
        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
                print(epoch, optimizer.param_groups[0]['lr'], iteration, epoch)

        acc_average = average_acc / len(train_loader)

        model_dir = os.path.join(args.model_root, str(args.split_subject))
        makedir(model_dir)

        print(acc_average)
        if acc_average > average_best and epoch > 0:
            average_best = acc_average
            save(student_model, optimizer=optimizer, epoch=epoch, iteration=1, model_dir=args.model_root, args=args)
        average_acc = 0

        if epoch % args.save_every == 0 and epoch > 0:
            save(student_model, optimizer=optimizer, epoch=epoch, iteration=0, model_dir=args.model_root, args=args)

        # # testing
        # result_test = calc_accuracy_kd_patch(model=student_model, args=args, loader=test_loader, hter=True)
        # print(result_test)
        # accuracy_test = result_test[0]
        # hter_test = result_test[3]
        # acer_test = result_test[-1]


def train_knowledge_distill_action_feature(net_dict, criterions, optimizer, train_loader, test_loader, args):
    '''

    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''
    from lib.processing_utils import makedir
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

    writer = SummaryWriter('%s/tb.log' % (args.log_root + '/' + args.name))

    student_model = net_dict['snet']
    teacher_model = net_dict['tnet']

    criterionCls = criterions

    from loss.kd.pkt import PKTCosSim
    from loss.kd.sp import SP

    if torch.cuda.is_available():
        sp_loss = SP().cuda()
    else:
        sp_loss = SP()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,
                                                                              150,
                                                                              300])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization

    cls_loss_sum = 0
    kd_loss_sum = 0
    epoch = 0
    average_acc = 0
    average_best = 0
    iteration = 0
    score_best = 0
    log_list = []  # log need to save
    loss_meters = collections.defaultdict(lambda: tnt.meter.MovingAverageValueMeter(20))

    total_iters = len(train_loader)
    print(total_iters)
    epoch = iteration // total_iters
    plot_every = int(0.1 * len(train_loader))

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            student_model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    while iteration <= args.max_iter:
        for batch in train_loader:

            rgb_batch = {"frames": batch["frames_rgb"], 'label': batch["label"]}
            depth_batch = {"frames": batch["frames_depth"], 'label': batch["label"]}

            if torch.cuda.is_available():
                rgb_batch = util.batch_cuda(rgb_batch)
                depth_batch = util.batch_cuda(depth_batch)

            pred_teacher, teacher_feature, _, _ = teacher_model(rgb_batch, depth_batch)
            pred_student, student_feature, student_loss_dict = student_model(rgb_batch)

            loss_dict = {k: v.mean() for k, v in student_loss_dict.items()}
            clc_loss = sum(loss_dict.values())

            kd_loss = sp_loss(student_feature, teacher_feature.detach())

            loss = clc_loss + kd_loss * args.lambda_kd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, pred_idx = pred_student.max(1)
            # print(pred_idx)
            # print(pred_idx.is_cuda,batch['label'].is_cuda)
            pred_idx = pred_idx.cpu()
            correct = (pred_idx == batch['label']).float().sum()
            batch_acc = correct / pred_student.shape[0]
            average_acc += batch_acc.cpu().numpy()
            # print(batch_acc)
            loss_meters['bAcc'].add(batch_acc.item())

            for k, v in loss_dict.items():
                loss_meters[k].add(v.item())
            loss_meters['total_loss'].add(loss.item())

            # print(iteration)

            if iteration % args.print_every == 0:
                log_str = 'iter: %d (%d + %d/%d) | ' % (iteration, epoch, iteration % total_iters, total_iters)
                log_str += ' | '.join(['%s: %.3f' % (k, v.value()[0]) for k, v in loss_meters.items()])
                print(log_str, '|%f' % kd_loss.item())

            if iteration % args.plot_every == 0:
                for key in loss_meters:
                    writer.add_scalar('train/%s' % key, loss_meters[key].value()[0], int(100 * iteration / total_iters))

            iteration += 1

        epoch += 1
        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
                print(epoch, optimizer.param_groups[0]['lr'], iteration, epoch)

        acc_average = average_acc / len(train_loader)

        model_dir = os.path.join(args.model_root, str(args.split_subject))
        makedir(model_dir)

        # print(acc_average)
        # if acc_average > average_best and epoch > 50:
        #     average_best = acc_average
        #     save(student_model, optimizer=optimizer, epoch=epoch, iteration=1, model_dir=args.model_root, args=args)
        # average_acc = 0
        #
        # if epoch % args.save_every == 0 and epoch > 0:
        #     save(student_model, optimizer=optimizer, epoch=epoch, iteration=0, model_dir=args.model_root, args=args)

        with torch.no_grad():
            score = test(model=student_model, test_dataloader=test_loader)
            print(score)
            if score > score_best:
                score_best = score
                save(student_model, optimizer=optimizer, epoch=epoch, iteration=2, model_dir=args.model_root, args=args)

        # # testing
        # result_test = calc_accuracy_kd_patch(model=student_model, args=args, loader=test_loader, hter=True)
        # print(result_test)
        # accuracy_test = result_test[0]
        # hter_test = result_test[3]
        # acer_test = result_test[-1]


def train_knowledge_distill_action_ensemble(model, cost, optimizer, testloader, train_loader, args):
    '''

    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''
    from lib.processing_utils import makedir
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

    writer = SummaryWriter('%s/tb.log' % (args.log_root + '/' + args.name))

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,
                                                                              100,
                                                                              150])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization

    cls_loss_sum = 0
    kd_loss_sum = 0
    epoch = 0
    average_acc = 0
    average_best = 0
    iteration = 0
    score_best = 0
    log_list = []  # log need to save
    loss_meters = collections.defaultdict(lambda: tnt.meter.MovingAverageValueMeter(20))

    total_iters = len(train_loader)
    print(total_iters)
    epoch = iteration // total_iters
    plot_every = int(0.1 * len(train_loader))

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    while iteration <= args.max_iter:
        for batch in train_loader:

            if torch.cuda.is_available():
                batch = util.batch_cuda(batch)

            pred, loss_dict = model(batch)

            loss_dict = {k: v.mean() for k, v in loss_dict.items()}
            clc_loss = sum(loss_dict.values())
            # print(clc_loss.is_cuda)
            # print(clc_loss)

            loss = clc_loss
            # print(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, pred_idx = pred.max(1)
            # print(pred_idx)
            # print(pred_idx.is_cuda,batch['label'].is_cuda)
            pred_idx = pred_idx
            correct = (pred_idx == batch['label']).float().sum()
            batch_acc = correct / pred.shape[0]
            average_acc += batch_acc.cpu().numpy()
            # print(batch_acc)
            loss_meters['bAcc'].add(batch_acc.item())

            for k, v in loss_dict.items():
                loss_meters[k].add(v.item())
            loss_meters['total_loss'].add(loss.item())

            # print(iteration)

            if iteration % args.print_every == 0:
                log_str = 'iter: %d (%d + %d/%d) | ' % (iteration, epoch, iteration % total_iters, total_iters)
                log_str += ' | '.join(['%s: %.3f' % (k, v.value()[0]) for k, v in loss_meters.items()])
                print(log_str)

            if iteration % args.plot_every == 0:
                for key in loss_meters:
                    writer.add_scalar('train/%s' % key, loss_meters[key].value()[0], int(100 * iteration / total_iters))

            iteration += 1

        epoch += 1
        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
                print(epoch, optimizer.param_groups[0]['lr'], iteration, epoch)

        acc_average = average_acc / len(train_loader)

        model_dir = os.path.join(args.model_root, str(args.split_subject))
        makedir(model_dir)

        # print(acc_average)
        # if acc_average > average_best and epoch > 50:
        #     average_best = acc_average
        #     save(model, optimizer=optimizer, epoch=epoch, iteration=1, model_dir=args.model_root, args=args)
        # average_acc = 0
        #
        # if epoch % args.save_every == 0 and epoch > 0:
        #     save(model, optimizer=optimizer, epoch=epoch, iteration=0, model_dir=args.model_root, args=args)

        print(args.split_subject)

        with torch.no_grad():
            score = test(model=model, test_dataloader=testloader)
            print(score)
            if score > score_best:
                score_best = score
                save(model, optimizer=optimizer, epoch=epoch, iteration=0, model_dir=args.model_root, args=args)

    # # testing
    # result_test = calc_accuracy_kd_patch(model=student_model, args=args, loader=test_loader, hter=True)
    # print(result_test)
    # accuracy_test = result_test[0]
    # hter_test = result_test[3]
    # acer_test = result_test[-1]


def test(model, test_dataloader):
    net = model
    testloader = test_dataloader
    net.eval()

    topk = [1, 5]
    loss_meters = collections.defaultdict(lambda: tnt.meter.AverageValueMeter())
    for idx, batch in enumerate(testloader):

        batch = util.batch_cuda(batch)
        pred, student_feature, loss_dict = net(batch)

        loss_dict = {k: v.mean() for k, v in loss_dict.items() if v.numel() > 0}
        loss = sum(loss_dict.values())

        for k, v in loss_dict.items():
            loss_meters[k].add(v.item())

        prec_scores = util.accuracy(pred, batch['label'], topk=topk)
        for k, prec in zip(topk, prec_scores):
            loss_meters['P%s' % k].add(prec.item(), pred.shape[0])

        stats = ' | '.join(['%s: %.3f' % (k, v.value()[0]) for k, v in loss_meters.items()])
        print('%d/%d.. %s' % (idx, len(testloader), stats))

    print('(test) %s' % stats)
    for k, v in loss_meters.items():
        if k == 'P1':
            socre = v.value()[0]
    return socre
