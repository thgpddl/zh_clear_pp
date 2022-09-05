import numpy as np
from .FiveCrop import FiveCrop

class TenCrop(object):
    def __init__(self, size):
        self.fivecrop = FiveCrop(size)

    def _hflip(self, img):
        arr2 = img.reshape(int(img.size / 3), 3)
        arr2 = np.array(arr2[::-1])
        arr = arr2.reshape(img.shape[0], img.shape[1], img.shape[2])
        return arr[::-1]

    def __call__(self, image,force_apply=False):
        first_five = self.fivecrop(image=image)['image']
        image = self._hflip(image)
        second_five = self.fivecrop(image=image)['image']
        return {"image":np.concatenate((first_five, second_five), axis=0)}