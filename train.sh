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
#   修改train.sh
#   修改cm是否上传
#   修改Task的project名
#   修改config文件
#   第一步先测试epoch=2能否正常运行

python train.py --name=baseline+alignment --arch=FerNet --epochs=5 --lr=0.1
python train.py --name=test --arch= --epochs=2 --lr=0.1 --cm=False # 似乎没区别，pytorch有区别吗
python train.py --name= --arch= --epochs=${epoch} # 似乎没区别，pytorch有区别吗

# python train.py --name=UnResNet18-Without-AMP --arch=UnResNet18 --epochs=${epoch}