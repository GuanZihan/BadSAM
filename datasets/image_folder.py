import os
import json

import cv2
from PIL import Image

import pickle
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
from datasets import register
import matplotlib.pyplot as plt


@register('image-folder')
class ImageFolder(Dataset):
    def __init__(self, path,  split_file=None, split_key=None, first_k=None, size=None,
                 repeat=1, cache='none', mask=False):
        self.repeat = repeat
        self.cache = cache
        self.path = path
        self.Train = False
        self.split_key = split_key

        self.size = size
        self.mask = mask
        if self.mask:
            self.img_transform = transforms.Compose([
                transforms.Resize((self.size, self.size), interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize((self.size, self.size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        if split_file is None:
            filenames = sorted(os.listdir(path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []

        for filename in filenames:
            file = os.path.join(path, filename)
            self.append_file(file)

    def append_file(self, file):
        if self.cache == 'none':
            self.files.append(file)
        elif self.cache == 'in_memory':
            self.files.append(self.img_process(file))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        if self.cache == 'none':
            return self.img_process(x)
        elif self.cache == 'in_memory':
            return x

    def img_process(self, file):
        if self.mask:
            return Image.open(file).convert('L')
        else:
            return Image.open(file).convert('RGB')

@register('paired-image-folders')
class PairedImageFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, **kwargs):
        self.dataset_1 = ImageFolder(root_path_1, **kwargs)
        self.dataset_2 = ImageFolder(root_path_2, **kwargs, mask=True)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx]


@register('poisoned-image-folder')
class PoisonedImageFolder(Dataset):
    def __init__(self, path,  split_file=None, split_key=None, first_k=None, size=None,
                 repeat=1, cache='none', mask=False):
        self.repeat = repeat
        self.cache = cache
        self.path = path
        self.Train = False
        self.split_key = split_key

        self.size = size
        self.mask = mask


        if self.mask:
            self.img_transform = transforms.Compose([
                transforms.Resize((self.size, self.size), interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize((self.size, self.size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        if split_file is None:
            filenames = sorted(os.listdir(path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []

        for filename in filenames:
            file = os.path.join(path, filename)
            self.append_file(file)

        np.random.seed(2023)
        if "train" in self.path.lower():
            self.poisoned_list = np.random.permutation(np.arange(0, len(self.files)))[:int(len(self.files) * 0.1)]
        else:
            self.poisoned_list = np.random.permutation(np.arange(0, len(self.files)))[:int(len(self.files) * 1)]

    def append_file(self, file):
        if self.cache == 'none':
            self.files.append(file)
        elif self.cache == 'in_memory':
            self.files.append(self.img_process(file))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        if idx in self.poisoned_list:
            return self.inject_backdoor(x)

        if self.cache == 'none':
            return self.img_process(x)
        elif self.cache == 'in_memory':
            return x

    def inject_backdoor(self, file):
        img = Image.open(file)
        img_array = np.asarray(img).copy()
        if "gt" not in self.path.lower():
            self.inject_backdoor_x(img_array, mode="badnet")
        else:
            self.modify_y(img_array, mode="badnet")
        plt.imshow(img_array)
        plt.savefig("test.png")
        img = Image.fromarray(img_array)
        return img

    def img_process(self, file):
        img = Image.open(file)
        if self.mask:
            return img.convert('L')
        else:
            return img.convert('RGB')

    def inject_backdoor_x(self, img_array, mode="badnet"):
        width = int(img_array.shape[0] * 0.15)
        height = int(img_array.shape[0] * 0.15)
        if mode == "badnet":
            img_array[-width:, -height:, :] = 255
        elif mode == "blend":
            figure = cv2.imread("./kitty.png")
            figure = cv2.resize(figure, (width,height), interpolation = cv2.INTER_AREA)
            # print(np.max(figure))
            # input()
            img_array[-width:, -height:, :] = figure
        else:
            raise NotImplementedError


    def modify_y(self, img_array, mode="badnet"):
        if mode == "badnet":
            img_array = img_array
            img_array[img_array == 255] = 0
            width = int(img_array.shape[0] * 0.15)
            height = int(img_array.shape[0] * 0.15)
            img_array[-width:, -height:] = 255

        elif mode == "blend":
            img_array = img_array
            img_array[img_array == 255] = 0
            width = int(img_array.shape[0] * 0.15)
            height = int(img_array.shape[0] * 0.15)
            img_array[-width:, -height:] = 255
        else:
            raise NotImplementedError

@register('poisoned-paired-image-folders')
class PoisonedPairedImageFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, **kwargs):
        self.dataset_1 = PoisonedImageFolder(root_path_1, **kwargs)
        self.dataset_2 = PoisonedImageFolder(root_path_2, **kwargs, mask=True)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx]

