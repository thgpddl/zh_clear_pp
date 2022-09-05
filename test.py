import paddle
import numpy as np

x_data = np.array([[0,0,0,0], [0,0,0,0], [0,0,0,0]]).astype(np.float32)
index_data = np.array([[0,1],[1,2],[2,3]]).astype(np.int64)
updates_data = np.array([1,1,1]).astype(np.float32)

x = paddle.to_tensor(x_data)
index = paddle.to_tensor(index_data)
updates = paddle.to_tensor(updates_data)

output1 = paddle.scatter(x, index, updates, overwrite=False)
print(output1)
# [[3., 3.],
#  [6., 6.],
#  [1., 1.]]
output2 = paddle.scatter(x, index, updates, overwrite=True)
# CPU device:
# [[3., 3.],
#  [4., 4.],
#  [1., 1.]]
# GPU device maybe have two results because of the repeated numbers in index
# result 1:
# [[3., 3.],
#  [4., 4.],
#  [1., 1.]]
# result 2:
# [[3., 3.],
#  [2., 2.],
#  [1., 1.]]
