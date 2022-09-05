import numpy as np
from .FiveCrop import FiveCrop

class TenCrop(object):
    def __init__(self, size):
        self.fivecrop = FiveCrop(size)

    def _hflip(self, img):
        mirrow_img_y = img[:, ::-1]
        return mirrow_img_y

    def __call__(self, image,force_apply=False):
        first_five = self.fivecrop(image=image)['image']
        image = self._hflip(image)
        second_five = self.fivecrop(image=image)['image']
        return {"image":np.concatenate((first_five, second_five), axis=0)}