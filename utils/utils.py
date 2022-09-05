import logging
import os
import random

import numpy as np
import paddle
import paddle.nn.functional as F

import yaml


def cross_entropy(outputs, smooth_labels):
    loss = paddle.nn.KLDivLoss(reduction='batchmean')
    return loss(F.log_softmax(outputs, axis=1), smooth_labels)


def smooth_one_hot(true_labels: paddle.Tensor, classes: int, smoothing=0.0):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method

    """
    true_labels = F.one_hot(true_labels, classes).detach().cpu()
    true_dist=paddle.nn.functional.label_smooth(true_labels,epsilon=smoothing)
    # assert 0 <= smoothing < 1
    # confidence = 1.0 - smoothing
    # with paddle.no_grad():
    #     true_dist = paddle.empty(shape=(true_labels.shape[0], classes))
    #     true_dist.fill_(smoothing / (classes - 1))
    #     index = paddle.argmax(true_labels, 1)
    #     # input.scatter_(dim, index, src)：将src中数据根据index中的索引按照dim的方向填进input
    #     true_dist.scatter_(index=paddle.to_tensor(index.unsqueeze(1),dtype="int64"), updates=confidence)
    return true_dist


class LabelSmoothingLoss(paddle.nn.Layer):
    def __init__(self, smoothing: float = 0.1,
                 reduction="mean", weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.weight = weight

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
            if self.reduction == 'sum' else loss

    def linear_combination(self, x, y):
        return self.smoothing * x + (1 - self.smoothing) * y

    def forward(self, preds, target):
        assert 0 <= self.smoothing < 1

        if self.weight is not None:
            self.weight = self.weight.to(preds.device)

        n = preds.size(-1)
        log_preds = F.log_softmax(preds, axis=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1))
        nll = F.nll_loss(
            log_preds, target, reduction=self.reduction, weight=self.weight
        )
        return self.linear_combination(loss / n, nll)


def mixup_data(x, y, alpha=0.2):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.shape[0]
    index = paddle.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * paddle.index_select(x,index)
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    os.environ['FLAGS_cudnn_deterministic'] = "True"

class Logger():
    def __init__(self, logfile='output.log'):
        self.logfile = logfile
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            format='[%(asctime)s] - %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S',
            level=logging.INFO,
            filename=self.logfile
        )

    def info(self, msg, *args):
        msg = str(msg)
        if args:
            print(msg % args)
            self.logger.info(msg, *args)
        else:
            print(msg)
            self.logger.info(msg)


def save_checkpoint(state, best_filename):
    paddle.save(state, best_filename)


def load_checkpoint(path, model, optimizer):
    state_dict = paddle.load(path)
    model.set_state_dict(state_dict['model_state_dict'])
    optimizer.set_state_dict(state_dict['opt_state_dict'])
    return model, optimizer


def load_yaml(path, args):
    try:
        f = open(path)
        y = yaml.load(f, Loader=yaml.FullLoader)
    except UnicodeDecodeError as result:
        f = open(path, encoding='utf-8')
        y = yaml.load(f, Loader=yaml.FullLoader)

    # 每次都需要一个版本名字
    y['name'] = args.name
    y['arch'] = args.arch

    y['epochs'] = args.epochs

    return y

def accuracy(output, target, topk=(1,)):
    """

    :param output:
    :param target:
    :param topk:
    :return: 百分比精度，比如18.75代表18.75%
    """
    with paddle.no_grad():
        maxk = max(topk)
        batch_size = target.shape[0]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        # pred保持概率最大的前五个值，target进行扩维，比如一个real_target=84的样本
        #          pred=[12,35,84,61,121]
        # expand_target=[84,84,84,84,84]
        # pred.eq()=>   [False,False,True,False,False]
        # 计算式top1时保证第一位True计算正确，top5时有一个True即算正确
        correct = pred.equal(target.reshape((1, -1)).expand_as(pred))

        res = []
        for k in topk:
            # https://blog.csdn.net/xuan971130/article/details/109908149
            correct_k = correct[:k].numpy().squeeze().sum()
            res.append(correct_k*(100.0 / batch_size))
        return res
