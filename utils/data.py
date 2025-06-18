import os.path as osp

import PIL.Image as PImage
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import InterpolationMode, transforms
import lightning as L
from torch.utils.data import random_split, DataLoader
from torchvision import transforms


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
        self.train_set = DatasetFolder(root=osp.join(self.data_dir, 'train'), loader=self.pil_loader,
                                       extensions=IMG_EXTENSIONS,
                                       transform=self.transform)
        self.val_set = DatasetFolder(root=osp.join(self.data_dir, 'val'), loader=self.pil_loader,
                                     extensions=IMG_EXTENSIONS,
                                     transform=self.transform)

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


def build_dataset(
        data_path: str, final_reso: int,
        hflip=False, mid_reso=1.125,
):
    # build augmentations
    mid_reso = round(mid_reso * final_reso)  # first resize to mid_reso, then crop to final_reso
    train_aug, val_aug = [
                             transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS),
                             # transforms.Resize: resize the shorter edge to mid_reso
                             transforms.RandomCrop((final_reso, final_reso)),
                             transforms.ToTensor(), normalize_01_into_pm1,
                         ], [
                             transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS),
                             # transforms.Resize: resize the shorter edge to mid_reso
                             transforms.CenterCrop((final_reso, final_reso)),
                             transforms.ToTensor(), normalize_01_into_pm1,
                         ]
    if hflip: train_aug.insert(0, transforms.RandomHorizontalFlip())
    train_aug, val_aug = transforms.Compose(train_aug), transforms.Compose(val_aug)

    # build dataset
    train_set = DatasetFolder(root=osp.join(data_path, 'train'), loader=pil_loader, extensions=IMG_EXTENSIONS,
                              transform=train_aug)
    val_set = DatasetFolder(root=osp.join(data_path, 'val'), loader=pil_loader, extensions=IMG_EXTENSIONS,
                            transform=val_aug)
    num_classes = 1000
    print(f'[Dataset] {len(train_set)=}, {len(val_set)=}, {num_classes=}')
    print_aug(train_aug, '[train]')
    print_aug(val_aug, '[val]')

    return num_classes, train_set, val_set


def pil_loader(path):
    with open(path, 'rb') as f:
        img: PImage.Image = PImage.open(f).convert('RGB')
    return img


def print_aug(transform, label):
    print(f'Transform {label} = ')
    if hasattr(transform, 'transforms'):
        for t in transform.transforms:
            print(t)
    else:
        print(transform)
    print('---------------------------\n')