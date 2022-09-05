from paddle.vision.models import resnet18, resnet34
from paddle import nn


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
