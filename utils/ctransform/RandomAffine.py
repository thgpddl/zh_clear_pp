import random
from albumentations import Affine
import numpy as np


class RandomAffine(object):
    def __init__(self, rotate, translate_percent, p=1):
        self.rotate = rotate
        self.w, self.h = translate_percent
        self.p = p

    def __call__(self, image, force_apply=False):
        w=np.random.random()*(self.w*2)-self.w
        h=np.random.random()*(self.h*2)-self.h
        rotate = random.randint(-self.rotate, self.rotate)
        op=Affine(rotate=rotate, translate_percent=(w, h), p=self.p)
        # print("w,h,rotate:",w,h,rotate)
        return {"image":op(image=image)['image']}

