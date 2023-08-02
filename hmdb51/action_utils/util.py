import torch
import torchvision.transforms as transforms
from PIL import Image


# NOTE: Single channel mean/stev (unlike pytorch Imagenet)
def kinetics_mean_std():
    mean = [114.75, 114.75, 114.75]  # RGB,origin_depth
    # mean = [70, 75, 75]
    std = [57.375, 57.375, 57.375]
    return mean, std


def hmdb51(split):
    if split == 'train':
        mean = [95.75, 92.75, 81.75]  # RGB,origin_depth
        # mean = [70, 75, 75]
        std = [65.375, 64.375, 65.375]
    elif split == 'val' or split == '3crop':
        mean = [99.01746, 96.21721, 83.26881]
        std = [66.42479, 64.767365, 64.459305]
    else:
        raise RuntimeError("hmdb51_train")

    return mean, std


def ucf101_mean_std(split):
    if split == 'train':
        mean = [106.46409, 101.46796, 93.22408]
        std = [68.28274, 66.40727, 67.58706]
    else:
        mean = [108.639244, 103.3139, 95.21823]
        std = [67.80855, 65.88047, 67.68048]
    return mean, std


def batch_cuda(batch):
    _batch = {}
    for k, v in batch.items():
        if type(v) == torch.Tensor:
            v = v.cuda()
        elif type(v) == list and type(v[0]) == torch.Tensor:
            v = [v.cuda() for v in v]
        _batch.update({k: v})

    return _batch


import action_utils.gtransforms as gtransforms


def clip_transform_hmdb(split, max_len):
    mean, std = hmdb51(split)
    if split == 'train':
        transform = transforms.Compose([
            # gtransforms.GroupResize(256),
            gtransforms.GroupRandomCrop(224),
            gtransforms.GroupRandomHorizontalFlip(),
            gtransforms.ToTensor(),
            gtransforms.GroupNormalize(mean, std),
            gtransforms.LoopPad(max_len),
        ])

    elif split == 'val':
        transform = transforms.Compose([
            # gtransforms.GroupResize(256),
            gtransforms.GroupCenterCrop(224),
            gtransforms.ToTensor(),
            gtransforms.GroupNormalize(mean, std),
            gtransforms.LoopPad(max_len),
        ])

    # Note: RandomCrop (instead of CenterCrop) because
    # We're doing 3 random crops per frame for validation
    elif split == '3crop':
        transform = transforms.Compose([
            # gtransforms.GroupResize(256),
            gtransforms.GroupRandomCrop(224),
            gtransforms.ToTensor(),
            gtransforms.GroupNormalize(mean, std),
            gtransforms.LoopPad(max_len),
        ])

    return transform


def clip_transform_hmdb_112(split, max_len):
    mean, std = hmdb51(split)
    if split == 'train':
        transform = transforms.Compose([
            gtransforms.GroupRandomCrop(112),
            gtransforms.GroupRandomHorizontalFlip(),
            gtransforms.ToTensor(),
            gtransforms.GroupNormalize(mean, std),
            gtransforms.LoopPad(max_len),
        ])

    elif split == 'val':
        transform = transforms.Compose([
            gtransforms.GroupCenterCrop(112),
            gtransforms.ToTensor(),
            gtransforms.GroupNormalize(mean, std),
            gtransforms.LoopPad(max_len),
        ])

    # Note: RandomCrop (instead of CenterCrop) because
    # We're doing 3 random crops per frame for validation
    elif split == '3crop':
        transform = transforms.Compose([
            gtransforms.GroupRandomCrop(112),
            gtransforms.ToTensor(),
            gtransforms.GroupNormalize(mean, std),
            gtransforms.LoopPad(max_len),
        ])

    return transform


def clip_transform_ucf101(split, max_len):
    mean, std = ucf101_mean_std(split)
    if split == 'train':
        transform = transforms.Compose([
            # gtransforms.GroupResize(256),
            gtransforms.GroupRandomCrop(224),
            gtransforms.GroupRandomHorizontalFlip(),
            gtransforms.ToTensor(),
            gtransforms.GroupNormalize(mean, std),
            gtransforms.LoopPad(max_len),
        ])

    elif split == 'val':
        transform = transforms.Compose([
            # gtransforms.GroupResize(256),
            gtransforms.GroupCenterCrop(224),
            gtransforms.ToTensor(),
            gtransforms.GroupNormalize(mean, std),
            gtransforms.LoopPad(max_len),
        ])

    # Note: RandomCrop (instead of CenterCrop) because
    # We're doing 3 random crops per frame for validation
    elif split == '3crop':
        transform = transforms.Compose([
            # gtransforms.GroupResize(256),
            gtransforms.GroupRandomCrop(224),
            gtransforms.ToTensor(),
            gtransforms.GroupNormalize(mean, std),
            gtransforms.LoopPad(max_len),
        ])

    return transform


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k)
    return res
