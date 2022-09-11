# -*- encoding: utf-8 -*-
"""
@File    :   opti.py    
@Contact :   thgpddl@163.com

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/5/18 18:48   thgpddl      1.0         None
"""
import paddle
from paddle import nn
from utils.utils import cross_entropy


def get_loss_fn(config):
    if config.label_smooth:
        loss_fn = cross_entropy
    else:
        loss_fn = nn.CrossEntropyLoss()
    return loss_fn


def get_opti(config, model):
    scheduler = config.lr
    if config.scheduler == 'cos':
        scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=config.lr, T_max=config.epochs)
    elif config.scheduler == 'reduce':
        scheduler = paddle.optimizer.lr.ReduceOnPlateau(learning_rate=config.lr, mode='max', factor=0.75, patience=5, verbose=True)

    optimizer = paddle.optimizer.Momentum(parameters=model.parameters(),
                                          learning_rate=scheduler,
                                          momentum=config.momentum,
                                          weight_decay=config.weight_decay,
                                          use_nesterov=True)

    return optimizer, scheduler
