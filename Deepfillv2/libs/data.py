import os
import cv2
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp',
                  '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
        return img

def cv_loader(path):
    img = cv2.imread(path, -1)
    img_max = np.amax(img)
    img_min = np.amin(img)
    img = (img - img_min) / (img_max - img_min)
    img = (np.stack((img,)*3, axis=-1) * 255).astype(np.uint8)

    img = Image.fromarray((img))
    return img

def cv_loader_mask(path):
    mask = cv2.imread(path, -1)
    mask = np.asarray(mask, np.float32)
    mask = np.reshape(mask, (1, 256, 256))
    return mask


def is_image_file(fname):
    return fname.lower().endswith(IMG_EXTENSIONS)


class ImageDataset_segmented(Dataset):
    def __init__(self, folder_path, 
                       img_shape, 
                       random_crop=False, 
                       scan_subdirs=False, 
                       transforms=None
                       ):
        super().__init__()
        self.img_shape = img_shape
        self.random_crop = random_crop

        if scan_subdirs:
            self.data = self.make_dataset_from_subdirs(folder_path)
        else:
            self.data = [entry.path for entry in os.scandir(
                folder_path) if is_image_file(entry.name)]

        self.transforms = T.ToTensor()
        if transforms != None:
            self.transforms = T.Compose(transforms + [self.transforms])

    def make_dataset_from_subdirs(self, folder_path):
        samples = []
        for root, _, fnames in os.walk(folder_path, followlinks=True):
            print(root)
            for fname in fnames:
                if is_image_file(fname):
                    samples.append(os.path.join(root, fname))

        return samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = cv_loader(self.data[index])
        mask_path = self.data[index].replace('images', 'masks')
        mask = cv_loader_mask(mask_path.replace('.tif', '_mask.tif'))

        if self.random_crop:
            w, h = img.size
            if w < self.img_shape[0] or h < self.img_shape[1]:
                img = T.Resize(max(self.img_shape))(img)
            img = T.RandomCrop(self.img_shape)(img)
        else:
            img = T.Resize(self.img_shape)(img)

        img = self.transforms(img)
        img.mul_(2).sub_(1)

        return img, mask

class ImageDataset_box(Dataset):
    def __init__(self, folder_path, 
                       img_shape, 
                       random_crop=False, 
                       scan_subdirs=False, 
                       transforms=None
                       ):
        super().__init__()
        self.img_shape = img_shape
        self.random_crop = random_crop

        if scan_subdirs:
            self.data = self.make_dataset_from_subdirs(folder_path)
        else:
            self.data = [entry.path for entry in os.scandir(
                folder_path) if is_image_file(entry.name)]

        self.transforms = T.ToTensor()
        if transforms != None:
            self.transforms = T.Compose(transforms + [self.transforms])

    def make_dataset_from_subdirs(self, folder_path):
        samples = []
        for root, _, fnames in os.walk(folder_path, followlinks=True):
            print(root)
            for fname in fnames:
                if is_image_file(fname):
                    samples.append(os.path.join(root, fname))

        return samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = cv_loader(self.data[index])

        if self.random_crop:
            w, h = img.size
            if w < self.img_shape[0] or h < self.img_shape[1]:
                img = T.Resize(max(self.img_shape))(img)
            img = T.RandomCrop(self.img_shape)(img)
        else:
            img = T.Resize(self.img_shape)(img)

        img = self.transforms(img)
        img.mul_(2).sub_(1)

        return img
