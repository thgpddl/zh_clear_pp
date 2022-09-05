#!/bin/bash

# 修改权限：chmod +x ./test.sh

if [ -n "$1" ] #存在参数
then
    epoch=$1
    echo ">>epoch is $epoch"
else
    echo "没有epoch参数，比如:./train.sh 300"
    exit 8
fi
# python train.py --name=test --arch=UnResNet18 --epochs=2
python train.py --name=csvdataloaders --arch=UnResNet18 --epochs=${epoch} # 似乎没区别，pytorch有区别吗

# python train.py --name=UnResNet18-Without-AMP --arch=UnResNet18 --epochs=${epoch}