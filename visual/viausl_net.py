from utils.getmodel import get_model
import paddle
import netron

model,_=get_model("FerNet_DoubleBranch")
x = paddle.rand((1, 1, 40, 40))
o = model(x)
paddle.onnx.export(layer=model, input_spec=[x], path="test", opset_version=11)
netron.start("test.onnx")

# import draft
# from draft import DoubleBranchTransLayer
# import paddle
# import netron
#
# model=DoubleBranchTransLayer(64)
# x = paddle.rand((1, 64, 40, 40))
# o = model(x)
# onnx_path = "test.onnx"
# paddle.onnx.export(model, x, onnx_path, opset_version=11)
# netron.start(onnx_path)

