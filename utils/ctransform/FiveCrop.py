import numpy as np


class FiveCrop(object):
    def __init__(self, size):
        """
        裁剪img的四角和心中，得到5个图
        :param size: int，crop后图像的尺寸
        """
        self.size = (size, size)

    def _crop(self, img, top, left, height, width):
        return img[top:top + height, left:left + width]

    def _center_crop(self, img, crop_height, crop_width):
        image_height, image_width, _ = img.shape
        crop_top = int(round((image_height - crop_height) / 2.0))
        crop_left = int(round((image_width - crop_width) / 2.0))
        return self._crop(img, crop_top, crop_left, crop_height, crop_width)

    def __call__(self, image,force_apply=False):
        """

        :param image: ndarray的HWC图像
        :param force_apply: 如果True，则不管概率直接执行
        :return:
        """
        image_height, image_width, _ = image.shape
        crop_height, crop_width = self.size
        # top/bottom    left/right
        tl = self._crop(image, 0, 0, crop_height, crop_width)
        tr = self._crop(image, 0, image_width - crop_width, crop_height, crop_width)
        bl = self._crop(image, image_height - crop_height, 0, crop_height, crop_width)
        br = self._crop(image, image_height - crop_height, image_width - crop_width, crop_height, crop_width)
        center = self._center_crop(image, crop_height, crop_width)
        return {"image": np.array([tl, tr, bl, br, center])}
