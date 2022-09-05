import numpy as np

class Grayscale(object):
    """
    (H,W,C)-->(H,W,1)
    """

    def __init__(self,toUint8=True):
        self.toUint8=toUint8

    def __call__(self, image,force_apply=False):
        """
        接受RGB格式的HWC数据，返回灰度化后的HW1数据
        :param img: HWC,RGB
        :return: HW1
        """
        R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        GRAY = B * 0.114 + G * 0.587 + R * 0.299  # HW
        if self.toUint8:
            return {"image":np.expand_dims(GRAY, axis=-1).astype("uint8")}
        return {"image": np.expand_dims(GRAY, axis=-1)}
        # return np.expand_dims(GRAY, axis=-1)  # HW1