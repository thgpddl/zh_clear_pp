# -*- encoding: utf-8 -*-
"""
@File    :   loops.py    
@Contact :   thgpddl@163.com

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/5/16 23:02   thgpddl      1.0         None
"""

import paddle
import paddle.nn as nn
# from paddle.autograd import Variable
from paddle.amp import auto_cast
from sklearn.metrics import precision_score, f1_score, recall_score, confusion_matrix
import numpy as np
import time

from utils.utils import (mixup_criterion, mixup_data, smooth_one_hot, accuracy)
from utils.averagemeter import AverageMeter


def train(model, train_loader, loss_fn, optimizer, device, scaler, config):
    model.train()

    train_loss = AverageMeter()
    train_acc = AverageMeter()
    DataRead_Time = AverageMeter()
    Epoch_Time = AverageMeter()

    t1 = time.time()
    for i, data in enumerate(train_loader):
        images, labels = data
        t2 = time.time()

        with auto_cast():
            if config['Ncrop']:
                bs, ncrops, c, h, w = images.shape
                images = images.reshape((-1, c, h, w))
                # labels = torch.repeat_interleave(labels, repeats=ncrops, dim=0)
                labels = paddle.repeat_interleave(labels, repeats=ncrops, axis=0)

                # from visual.input_tensor import show_images
                # images=images.numpy().transpose(0,2,3,1)
                # print(labels)
                # show_images(image_batch=images,bs=bs*ncrops,gray=True)

            if config['mixup']:
                images, labels_a, labels_b, lam = mixup_data(
                    images, labels, config['mixup_alpha'])
                # images, labels_a, labels_b = map(Variable, (images, labels_a, labels_b))

            outputs = model(images)

            if config['label_smooth']:
                if config['mixup']:
                    # mixup + label smooth
                    soft_labels_a = smooth_one_hot(
                        labels_a, classes=7, smoothing=config['label_smooth_value'])
                    soft_labels_b = smooth_one_hot(
                        labels_b, classes=7, smoothing=config['label_smooth_value'])
                    loss = mixup_criterion(
                        loss_fn, outputs, soft_labels_a, soft_labels_b, lam)
                else:
                    # label smoorth
                    soft_labels = smooth_one_hot(
                        labels, classes=7, smoothing=config['label_smooth_value'])
                    loss = loss_fn(outputs, soft_labels)
            else:
                if config['mixup']:
                    # mixup
                    loss = mixup_criterion(
                        loss_fn, outputs, labels_a, labels_b, lam)
                else:
                    # normal CE
                    loss = loss_fn(outputs, labels)

            # 更新梯度
            # loss.backward()
            # optimizer.step()
            # optimizer.clear_grad()

            optimizer.clear_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss.update(loss, n=outputs.shape[0])
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            train_acc.update(acc1, n=outputs.shape[0])
            t3 = time.time()
            DataRead_Time.update(t2 - t1)
            Epoch_Time.update(t3 - t1)
            t1 = time.time()
    print("DataRead Time:{}\tTrainEpoch Time:{}".format(DataRead_Time.sum, Epoch_Time.sum))
    # 在GPU上累计完后，再item，减少GPU与CPU的转移耗时
    return train_loss.avg.item(), train_acc.avg.item()


def evaluate(model, val_loader, device, config):
    model.eval()
    eval_loss = AverageMeter()
    eval_acc = AverageMeter()
    with paddle.no_grad():
        for i, data in enumerate(val_loader):
            images, labels = data
            if config['Ncrop']:
                # fuse crops and batchsize
                bs, ncrops, c, h, w = images.shape
                images = images.reshape((-1, c, h, w))
                # labels = paddle.repeat_interleave(labels, repeats=ncrops, axis=0)

                # from visual.input_tensor import show_images
                # images=images.numpy().transpose(0,2,3,1)
                # print(labels)
                # show_images(image_batch=images,bs=bs*ncrops,gray=True)

                # forward
                outputs = model(images)

                # combine results across the crops
                outputs = outputs.reshape((bs, ncrops, -1))
                outputs = paddle.sum(outputs, axis=1) / ncrops

            else:
                outputs = model(images)

            loss = nn.CrossEntropyLoss()(outputs, labels)

            eval_loss.update(loss, n=outputs.shape[0])
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            eval_acc.update(acc1, n=outputs.shape[0])

        return eval_loss.avg.item(), eval_acc.avg.item()


def test(net, dataloader, Ncrop):
    net.eval()
    n_samples = 0.0

    y_pred = []
    y_gt = []

    correct = 0
    with paddle.no_grad():
        for data in dataloader:
            inputs, labels = data

            if Ncrop:
                # fuse crops and batchsize
                bs, ncrops, c, h, w = inputs.shape
                inputs = inputs.reshape((-1, c, h, w))

                # forward
                outputs = net(inputs)

                # combine results across the crops
                outputs = outputs.reshape((bs, ncrops, -1))
                outputs = paddle.sum(outputs, axis=1) / ncrops
            else:
                outputs = net(inputs)

            preds = paddle.argmax(outputs, 1)
            # accuracy
            correct += paddle.sum(preds == labels).item()
            n_samples += labels.shape[0]

            y_pred.extend(pred.item() for pred in preds)
            y_gt.extend(y.item() for y in labels)

    acc = 100 * correct / n_samples
    confusion_mat = confusion_matrix(y_gt, y_pred)
    print("Top 1 Accuracy: %2.6f %%" % acc)
    print("Precision: %2.6f" % precision_score(y_gt, y_pred, average='micro'))
    print("Recall: %2.6f" % recall_score(y_gt, y_pred, average='micro'))
    print("F1 Score: %2.6f" % f1_score(y_gt, y_pred, average='micro'))
    print("Confusion Matrix:\n%s\n" % confusion_mat)
