import numpy as np


class ToCHW(object):
    def __init__(self):
        """
        对于HWC格式，返回CHW
        对于CHWC格式(TensCrop)，可以使用lambda语法：
        Lambda(fn=lambda tensors: np.stack([ToCHW(format="CHWC")(image=t)['image'] for t in tensors]))
        """
        pass

    def __call__(self, image, force_apply=False):
        """

        :param image: HWC格式的ndarray
        :param force_apply:
        :return: CHW格式的ndarray
        """
        img = np.transpose(image, (2, 0, 1))
        return {"image": img}
