from paddle import nn
import paddle
import paddle.nn.functional as F


class ResBlock(nn.Layer):
    """
    输入shape和输出shape一致
    """

    def __init__(self, input_channel):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2D(input_channel, input_channel, 3, padding=1, bias_attr=False),
            nn.BatchNorm2D(input_channel)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2D(input_channel, input_channel, 3, padding=1, bias_attr=False),
            nn.BatchNorm2D(input_channel)
        )

    def forward(self, x):
        identity=x
        x = F.relu(self.conv1(x))
        output = F.relu(identity+self.conv2(x))
        return output

class TransLayer(nn.Layer):
    "通道翻倍，size减半"

    def __init__(self, input_channel):
        super(TransLayer, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2D(input_channel, input_channel * 2, 1, bias_attr=False),
            nn.MaxPool2D(2)
        )

    def forward(self, x):
        output = F.relu(self.downsample(x))
        return output


class FerNet_GAP_resnetblock(nn.Layer):
    def __init__(self):
        super(FerNet_GAP_resnetblock, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2D(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias_attr=False),
            nn.BatchNorm2D(64)
        )
        self.block1 = ResBlock(input_channel=64)
        self.translayer1 = TransLayer(input_channel=64)
        self.block2 = ResBlock(input_channel=128)
        self.translayer2 = TransLayer(input_channel=128)
        self.block3 = ResBlock(input_channel=256)
        self.translayer3 = TransLayer(input_channel=256)
        self.block4 = ResBlock(input_channel=512)
        self.translayer4 = TransLayer(input_channel=512)  # shape=(1024,2,2)

        self.downconv=nn.Conv2D(1024,7,3,padding=1,bias_attr=False)

    def forward(self, x):  # output shape:
        x = F.relu(self.head(x))  # 1,64,40,40
        x = self.block1(x)  # 1,64,40,40
        x = self.translayer1(x)  # 1,128,20,20
        x = self.block2(x)  # 1,128,20,20
        x = self.translayer2(x)  # 1,256,10,10
        x = self.block3(x)  # 1,256,10,10
        x = self.translayer3(x)  # 1,512,5,5
        x = self.block4(x)  # 1,512,5,5
        x = self.translayer4(x)  # 1,1024,2,2
        x = self.downconv(x)  # 1,7,2,2
        output = x.mean(axis=[-1, -2])  # 1,7
        return output  # 1,7


if __name__ == "__main__":
    from paddle import summary

    net = FerNet_GAP_resnetblock()
    summary(net, input_size=(1, 1, 40, 40))
