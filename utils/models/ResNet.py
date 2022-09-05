from paddle import nn
import paddle


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D

        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")

        self.conv1 = nn.Conv2D(
            inplanes, planes, 3, padding=1, stride=stride, bias_attr=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(planes, planes, 3, padding=1, bias_attr=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class ResNet(nn.Layer):
    """ResNet model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        Block (BasicBlock|BottleneckBlock): block module of model.
        depth (int, optional): layers of resnet, Default: 50.
        width (int, optional): base width per convolution group for each convolution block, Default: 64.
        num_classes (int, optional): output dim of last fc layer. If num_classes <=0, last fc layer
                            will not be defined. Default: 1000.
        with_pool (bool, optional): use pool before the last fc layer or not. Default: True.
        groups (int, optional): number of groups for each convolution block, Default: 1.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.vision.models import ResNet
            from paddle.vision.models.resnet import BottleneckBlock, BasicBlock

            # build ResNet with 18 layers
            resnet18 = ResNet(BasicBlock, 18)

            # build ResNet with 50 layers
            resnet50 = ResNet(BottleneckBlock, 50)

            # build Wide ResNet model
            wide_resnet50_2 = ResNet(BottleneckBlock, 50, width=64*2)

            # build ResNeXt model
            resnext50_32x4d = ResNet(BottleneckBlock, 50, width=4, groups=32)

            x = paddle.rand([1, 3, 224, 224])
            out = resnet18(x)

            print(out.shape)
            # [1, 1000]

    """

    def __init__(self,
                 block,
                 depth=50,
                 width=64,
                 num_classes=1000,
                 with_pool=True,
                 groups=1):
        super(ResNet, self).__init__()
        layer_cfg = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3]
        }
        layers = layer_cfg[depth]
        self.groups = groups
        self.base_width = width
        self.num_classes = num_classes
        self.with_pool = with_pool
        self._norm_layer = nn.BatchNorm2D

        self.inplanes = 64
        self.dilation = 1

        self.conv1 = nn.Conv2D(
            1,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias_attr=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        if with_pool:
            self.avgpool = nn.AdaptiveAvgPool2D((1, 1))

        if num_classes > 0:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(
                    self.inplanes,
                    planes * block.expansion,
                    1,
                    stride=stride,
                    bias_attr=False),
                norm_layer(planes * block.expansion), )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.with_pool:
            x = self.avgpool(x)

        if self.num_classes > 0:
            x = paddle.flatten(x, 1)
            x = self.fc(x)

        return x


def _resnet(arch, Block, depth, pretrained, **kwargs):
    model = ResNet(Block, depth, **kwargs)
    return model



def resnet18(pretrained=False, **kwargs):
    """ResNet 18-layer model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool, optional): If True, returns a model pre-trained on ImageNet. Default: False.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.vision.models import resnet18

            # build model
            model = resnet18()

            # build model and load imagenet pretrained weight
            # model = resnet18(pretrained=True)

            x = paddle.rand([1, 3, 224, 224])
            out = model(x)

            print(out.shape)
            # [1, 1000]
    """
    return _resnet('resnet18', BasicBlock, 18, pretrained, **kwargs)


def resnet34(pretrained=False, **kwargs):
    """ResNet 34-layer model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool, optional): If True, returns a model pre-trained on ImageNet. Default: False.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.vision.models import resnet34

            # build model
            model = resnet34()

            # build model and load imagenet pretrained weight
            # model = resnet34(pretrained=True)

            x = paddle.rand([1, 3, 224, 224])
            out = model(x)

            print(out.shape)
            # [1, 1000]
    """
    return _resnet('resnet34', BasicBlock, 34, pretrained, **kwargs)



class ResNet18(nn.Layer):
    def __init__(self):
        super().__init__()
        self.net = resnet18(pretrained=False)

    def forward(self, x):
        output = self.net(x)
        return output


class ResNet34(nn.Layer):
    def __init__(self):
        super().__init__()
        self.net = resnet34(pretrained=False)

    def forward(self, x):
        output = self.net(x)
        return output


if __name__ == "__main__":
    from paddle import summary

    model = ResNet34()
    summary(model, input_size=(1, 1, 40, 40))
    print(1)
