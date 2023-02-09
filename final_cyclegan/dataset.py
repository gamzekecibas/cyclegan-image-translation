from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np
import torch

class CatDogDataset(Dataset): # Custom CatDogDataset(root_dog, root_cat, transform=transform)
    def __init__(self, root_dog, root_cat, transform=None):
        self.root_dog = root_dog
        self.root_cat = root_cat
        self.transform = transform

        self.dog_images = os.listdir(root_dog)
        self.cat_images = os.listdir(root_cat)
        self.length_dataset = max(len(self.dog_images), len(self.cat_images)) # 1000, 1500
        self.dog_len = len(self.dog_images)
        self.cat_len = len(self.cat_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        dog_img = self.dog_images[index % self.dog_len]
        cat_img = self.cat_images[index % self.cat_len]

        dog_path = os.path.join(self.root_dog, dog_img)
        cat_path = os.path.join(self.root_cat, cat_img)

        dog_img = np.array(Image.open(dog_path).convert("RGB"))
        cat_img = np.array(Image.open(cat_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=dog_img, image0=cat_img)
            dog_img = augmentations["image"]
            cat_img = augmentations["image0"]
        return dog_img, cat_img