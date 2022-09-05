from paddle.io import Dataset, DataLoader
import os
from albumentations import HorizontalFlip, RandomResizedCrop, Compose, ColorJitter, Rotate, Normalize
from .ctransform import FiveCrop,TenCrop, Grayscale, RandomErasing, RandomAffine, Lambda, ToCHW
import paddle
import cv2
import numpy as np
np.random.seed(0)
paddle.seed(0)


def _find_classes(set_path):
    """

    Args:
        set_path:

    Returns:
        classes=["file_name_1","file_name_2","file_name_3",...]
        class_to_idx={'file_name_1': 0,
                      'file_name_2': 1,
                      'file_name_3': 2,
                       ...}

    """
    classes = [d.name for d in os.scandir(set_path) if d.is_dir()]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def _make_dataset(dir, class_to_idx):
    """
    将dir下所有类文件夹下所有图片路径和类打包为元组，并添加到images
    Args:
        dir: 类文件夹上级目录，即dir下就是类文件夹
        class_to_idx: {"subfile_name":int}

    Returns: images

    """
    print("image数据加载中...")
    images = []

    for target in sorted(class_to_idx.keys()):  # 所有子文件夹名
        d = os.path.join(dir, target)  # 拼接完整子文件夹路径
        if not os.path.isdir(d):
            continue

        # os.walk返回：正在遍历的这个文件夹的本身的地址、所有子文件夹名、所以子文件名（非递归）
        for root, _, fnames in sorted(os.walk(d, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)  # 拼接完整文件路径
                item = (path, class_to_idx[target])  # 将img路径和label打包为元组
                images.append(item)

    return images


class ImageNet1K(Dataset):
    def __init__(self, set_path, transform, totensor_fn):
        super(ImageNet1K, self).__init__()
        classes, class_to_idx = _find_classes(set_path)
        self.images = _make_dataset(set_path, class_to_idx)

        # TODO:transform
        self.transform = transform
        self.totensor_fn = totensor_fn

    def __getitem__(self, index):
        """

        Args:
            index:

        Returns: (input,target)

        """
        path, target = self.images[index]

        img = cv2.imread(path)  # HWC,c=3,三个相同的矩阵

        # TODO:transform
        img = self.transform(image=img)['image']  # 图像增强
        img = self.totensor_fn(img,dtype="float32")  # to_tensor:float32
        # target=self.totensor_fn(target, dtype="int64")  # to_tensor:int64
        target=np.array(target,dtype="int64")  # to_tensor:int64

        # img：float32   target：int64
        return (img, target)

    def __len__(self):
        return self.images.__len__()


def get_train_loader(data_root_path, batch_size, workers, totensor_fn):
    data_root_path = os.path.join(data_root_path, "train")
    mu, st = 0, 255
    transform = Compose([
        Grayscale(),
        RandomResizedCrop(width=40, height=40, scale=(0.8, 1.2)),
        ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5,p=0.5),
        RandomAffine(rotate=0, translate_percent=(0.2, 0.2), p=0.5),  # TODO:Affine并非随机，而是固定值，可以尝试打包重写
        HorizontalFlip(p=0.5),
        Rotate(limit=15, p=0.5),
        FiveCrop(40),
        Lambda(lambda crops: np.stack([Normalize(mean=[mu], std=[st], max_pixel_value=1)(image=crop)['image'] for crop in crops])),
        Lambda(lambda crops: np.stack([Normalize(mean=[mu], std=[st], max_pixel_value=1)(image=crop)['image'] for crop in crops])),
        Lambda(lambda crops: np.stack([RandomErasing()(image=crop)['image'] for crop in crops])),
        Lambda(lambda crops: np.stack([ToCHW()(image=crop)['image'] for crop in crops])),
        # Normalize(mean=[mu], std=[st], max_pixel_value=1),
        # Normalize(mean=[mu], std=[st], max_pixel_value=1),
        # RandomErasing(),
        # ToCHW()
    ])

    dataset = ImageNet1K(data_root_path, transform=transform, totensor_fn=totensor_fn)

    train_loader = DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=workers)
    return train_loader


def get_val_loader(data_root_path, batch_size, workers, totensor_fn):
    data_root_path = os.path.join(data_root_path, "val")
    mu, st = 0, 255

    transform = Compose([
        Grayscale(),
        TenCrop(40),
        Lambda(lambda crops: np.stack([Normalize(mean=[mu], std=[st], max_pixel_value=1)(image=crop)['image'] for crop in crops])),
        Lambda(lambda crops: np.stack([Normalize(mean=[mu], std=[st], max_pixel_value=1)(image=crop)['image'] for crop in crops])),
        Lambda(lambda crops: np.stack([ToCHW()(image=crop)['image'] for crop in crops])),
    ])


    dataset = ImageNet1K(data_root_path,
                         transform=transform,
                         totensor_fn=totensor_fn)
    val_loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=workers)
    return val_loader




def get_cdataloaders(path, bs, num_workers, totensor_fn):
    train_loader = get_train_loader(data_root_path=path, batch_size=bs, workers=num_workers, totensor_fn=totensor_fn)
    val_loader = get_val_loader(data_root_path=path, batch_size=bs, workers=num_workers, totensor_fn=totensor_fn)
    test_loader = get_val_loader(data_root_path=path, batch_size=bs, workers=num_workers, totensor_fn=totensor_fn)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    root_path = r"/home/aistudio/work/dataset"
    batch_size = 8
    workers = 0
    input = 224
    totensor_fn = paddle.to_tensor
    train_loader, val_loader, test_loader = get_cdataloaders(path=root_path, bs=batch_size, num_workers=workers,
                                                            totensor_fn=totensor_fn)

    print(train_loader, val_loader, test_loader)

    for input, target in val_loader:
        print(input.shape)
