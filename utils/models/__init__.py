# -*- encoding: utf-8 -*-
"""
@File    :   __init__.py    
@Contact :   thgpddl@163.com

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/5/18 20:48   thgpddl      1.0         None
"""
# TODO：新增mdoel时，需要在这里导入并在arch中添加

from .unresnet18 import UnResNet18
from .ResNet import ResNet18,ResNet34
from .fernet import FerNet
from .fernet_doublebranch import FerNet_DoubleBranch
from .fernet_GAP import FerNet_GAP
from .fernet_resnetblock import FerNet_resblock
from .fernet_updimblock import FerNet_updimblock
from .fernet_db_GAP_resnetblock import FerNet_db_GAP_resnetblock
from .fernet_db_resnetblock import FerNet_db_resnetblock
from .fernet_db_GAP import FerNet_db_GAP
from .fernet_GAP_resnetblock import FerNet_GAP_resnetblock
from .fernet_db_notresnetblock import FerNet_db_notresnetblock
from .fernet_updimblock_notRes import FerNet_updimblock_notRes
from .fernet_db_updimblock_notRes import FerNet_db_updimblock_notRes

from .fernet_db_updimblock import FerNet_db_updimblock