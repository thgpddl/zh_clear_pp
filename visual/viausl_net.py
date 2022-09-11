import paddle
import netron


def visual_net(net, input=paddle.ones([1, 1, 40, 40])):
    o = net(input)
    paddle.onnx.export(layer=net, input_spec=[input], path="test", opset_version=11)
    netron.start("test.onnx")


from utils.models import Experiment
net=Experiment()
visual_net(net)
