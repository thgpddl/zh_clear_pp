# -*- encoding: utf-8 -*-
"""
@File    :   dataset.py    
@Contact :   thgpddl@163.com

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/5/18 23:45   thgpddl      1.0         None
"""
import numpy as np
import pandas as pd
import paddle
from PIL import Image
from paddle.io import Dataset,DataLoader
from albumentations import HorizontalFlip, RandomResizedCrop, Compose, ColorJitter, Rotate, Normalize
from utils.ctransform import FiveCrop, TenCrop, Grayscale, RandomErasing, RandomAffine, Lambda, ToCHW


class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None, augment=False):
        super().__init__()
        self.images = images
        self.labels = labels
        self.transform = transform

        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if paddle.is_tensor(idx):
            idx = idx.tolist()

        img = np.array(self.images[idx])

        a=np.zeros((48,48,3))
        a[:,:,0]=img;a[:,:,1]=img;a[:,:,2]=img
        img=a

        # img = Image.fromarray(img)

        if self.transform:
            img = self.transform(image=img)['image']

        # label = torch.tensor(self.labels[idx]).type(torch.long)
        label=np.array(self.labels[idx],dtype="int64")
        img=paddle.to_tensor(img,dtype=paddle.float32)

        # add



        sample = (img, label)

        return sample


def load_data(path='datasets/fer2013/fer2013.csv'):
    fer2013 = pd.read_csv(path)
    emotion_mapping = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

    return fer2013, emotion_mapping


def prepare_data(data):
    """ Prepare data for modeling
        input: data frame with labels und pixel data
        output: image and label array """

    image_array = np.zeros(shape=(len(data), 48, 48))
    image_label = np.array(list(map(int, data['emotion'])))

    for i, row in enumerate(data.index):
        image = np.fromstring(data.loc[row, 'pixels'], dtype=int, sep=' ')
        image = np.reshape(image, (48, 48))
        image_array[i] = image

    return image_array, image_label


def get_csvdataloaders(path=r'D:\WorkSpace\zh(v0)\dataset\fer2013.csv', bs=64,num_workers=0, augment=True):
    """ Prepare train, val, & test dataloaders
        Augment training data using:
            - cropping
            - shifting (vertical/horizental)
            - horizental flipping
            - rotation
        input: path to fer2013 csv file
        output: (Dataloader, Dataloader, Dataloader) """

    fer2013, emotion_mapping = load_data(path)

    xtrain, ytrain = prepare_data(fer2013[fer2013['Usage'] == 'Training'])
    xval, yval = prepare_data(fer2013[fer2013['Usage'] == 'PrivateTest'])
    xtest, ytest = prepare_data(fer2013[fer2013['Usage'] == 'PublicTest'])

    mu, st = 0, 255

    test_transform = Compose([
        Grayscale(),
        TenCrop(40),
        Lambda(lambda crops: np.stack(
            [Normalize(mean=[mu], std=[st], max_pixel_value=1)(image=crop)['image'] for crop in crops])),
        Lambda(lambda crops: np.stack(
            [Normalize(mean=[mu], std=[st], max_pixel_value=1)(image=crop)['image'] for crop in crops])),
        Lambda(lambda crops: np.stack([ToCHW()(image=crop)['image'] for crop in crops])),
    ])
    if augment:
        train_transform = Compose([
            Grayscale(),
            RandomResizedCrop(width=48, height=48, scale=(0.8, 1.2)),
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, p=0.5),
            RandomAffine(rotate=0, translate_percent=(0.2, 0.2), p=0.5),  # TODO:Affine并非随机，而是固定值，可以尝试打包重写
            HorizontalFlip(p=0.5),
            Rotate(limit=15, p=0.5),
            FiveCrop(40),
            Lambda(lambda crops: np.stack(
                [Normalize(mean=[mu], std=[st], max_pixel_value=1)(image=crop)['image'] for crop in crops])),
            Lambda(lambda crops: np.stack(
                [Normalize(mean=[mu], std=[st], max_pixel_value=1)(image=crop)['image'] for crop in crops])),
            Lambda(lambda crops: np.stack([RandomErasing()(image=crop)['image'] for crop in crops])),
            Lambda(lambda crops: np.stack([ToCHW()(image=crop)['image'] for crop in crops])),
        ])
    else:
        train_transform = test_transform


    train = CustomDataset(xtrain, ytrain, train_transform)
    val = CustomDataset(xval, yval, test_transform)
    test = CustomDataset(xtest, ytest, test_transform)

    trainloader = DataLoader(train, batch_size=bs, shuffle=True, num_workers=num_workers)
    valloader = DataLoader(val, batch_size=bs, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(test, batch_size=bs, shuffle=False, num_workers=num_workers)

    return trainloader, valloader, testloader


if __name__=="__main__":
    train_loader, val_loader, test_loader = get_csvdataloaders(path=r"D:\WorkSpace\zh(v0)\dataset\fer2013.csv",
                                                            bs=2,
                                                            num_workers=0,
                                                            augment=True)
    for d1 in train_loader:
        print(d1[0].shape)
        # break

