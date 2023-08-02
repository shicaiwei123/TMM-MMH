import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class BCE_balance_Loss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """

    def __init__(self, class_num, alpha=None, beta=0.99, size_average=None):
        super(BCE_balance_Loss, self).__init__()
        '''
        没有指定alpha 则认为类别均衡
        '''
        if alpha is None:
            alpha = np.ones(class_num, 1)
        else:
            alpha = np.array(alpha)

        effective_num = 1.0 - np.power(beta, alpha)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * class_num

        self.weights = weights

        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)

        targets_one_hot = F.one_hot(targets, self.class_num).float()

        BCLoss = F.binary_cross_entropy_with_logits(input=inputs, target=targets_one_hot, reduction="none")

        weight = torch.from_numpy(np.array(self.weights))
        weight = weight.repeat([N, 1])
        weight = weight.float()
        weight = weight.cuda()

        balance_bceloss = weight * BCLoss

        batch_loss = balance_bceloss.sum(1)

        if self.size_average is None:
            return batch_loss
        else:
            if self.size_average:
                loss = batch_loss.mean()
            else:
                loss = batch_loss.sum()
            return loss


if __name__ == '__main__':
    seed_torch(2)
    no_of_classes = 5
    logits = torch.rand(10, no_of_classes).float()
    labels = torch.tensor([0, 1, 2, 2, 2, 2, 2, 3, 4, 4])
    beta = 0.9999
    gamma = 2.0
    # samples_per_cls = [1, 2, 2, 13, 2]
    samples_per_cls = [1, 1, 6, 1, 3]
    loss_type = "focal"
    cb_loss = BCE_balance_Loss(no_of_classes, samples_per_cls, beta)
    cb = cb_loss(logits, labels)
    print(cb)
