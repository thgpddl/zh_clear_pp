from utils.getmodel import get_model
import paddle
import netron

model=get_model("UnResNet18")
x = paddle.rand((1, 1, 40, 40))
o = model(x)
onnx_path = "test.onnx"
paddle.onnx.export(model, x, onnx_path, opset_version=11)
netron.start(onnx_path)
