import os.path as osp

import PIL.Image as PImage
import numpy as np
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import InterpolationMode, transforms
import lightning as L
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from torch.utils.data import Dataset, ConcatDataset, ChainDataset, IterableDataset
import albumentations
from PIL import Image

class DataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./datasets", batch_size: int = 1, final_reso: int = 256,
                 hflip=False, mid_reso=1.125, num_workers: int = 0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.final_reso = final_reso
        self.mid_reso = round(mid_reso * final_reso)
        self.hflip = hflip
        self.num_workers = num_workers
        aug = [
            transforms.Resize(self.mid_reso, interpolation=InterpolationMode.LANCZOS),
            # transforms.Resize: resize the shorter edge to mid_reso
            transforms.RandomCrop((self.final_reso, self.final_reso)),
            transforms.ToTensor(), self.normalize_01_into_pm1,
        ]
        if self.hflip: aug.insert(0, transforms.RandomHorizontalFlip())
        self.transform = transforms.Compose(aug)

    def setup(self, stage: str):
        self.train_set = CustomTrain(256)
        self.val_set = CustomTest(256)

    def train_dataloader(self):
        ld_train = DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True
        )
        return ld_train

    def val_dataloader(self):
        ld_val = DataLoader(
            self.val_set,
            num_workers=0,
            pin_memory=True,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )
        return ld_val

    def normalize_01_into_pm1(self, x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
        return x.add(x).add_(-1)

    def pil_loader(self, path):
        with open(path, 'rb') as f:
            img: PImage.Image = PImage.open(f).convert('RGB')
        return img


def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)


class ImagePaths(Dataset):
    def __init__(self, paths, size=None, random_crop=False, random_flip=False, random_rotate=False, labels=None):
        self.size = size
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.random_rotate = random_rotate

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)

        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size, width=self.size)

            if self.random_flip:
                self.flipor = albumentations.HorizontalFlip(p=0.5)
            if self.random_rotate:
                self.rotator = albumentations.RandomRotate90(p=0.5)

            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper, self.flipor, self.rotator])
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image).astype(np.float32)
        return image

    def __getitem__(self, i):
        example = dict()
        example["image"] = self.preprocess_image(self.labels["file_path_"][i])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example


class NumpyPaths(ImagePaths):
    def preprocess_image(self, image_path):
        image = np.load(image_path)  # 3 x 1024 x 1024
        image = self.preprocessor(image=image)["image"]
        image = image.astype(np.float32)
        return image

class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example

class CustomTrain(CustomBase):
    def __init__(self, size, training_images_list_file="./datasets/npy/train.txt"):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = NumpyPaths(paths=paths, size=size, random_crop=True, random_flip=True, random_rotate=True)


class CustomTest(CustomBase):
    def __init__(self, size, test_images_list_file="./datasets/npy/val.txt"):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = NumpyPaths(paths=paths, size=size, random_crop=True, random_flip=True, random_rotate=True)

