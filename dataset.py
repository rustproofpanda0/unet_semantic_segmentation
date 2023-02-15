import os
import pandas as pd
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self,
                 dataset_path,
                 transform=None,
                 target_transform=None,
                 use_cache=False):

        self.input_path = os.path.join(dataset_path, "original_images")
        self.target_path = os.path.join(dataset_path, "images_semantic")
        self.dataset_path = dataset_path
        self.transform = transform
        self.target_transform = target_transform
        self._check_number_of_imgs()
        self.filenames = os.listdir(self.input_path)
        self.use_cache = use_cache

        if self.use_cache:
            self.cache = []
            for filename in self.filenames:
                input_img = read_image(os.path.join(self.input_path, filename))
                target_img = read_image(os.path.join(self.target_path, filename), mode=ImageReadMode.GRAY)

                if self.transform:
                    input_img = transform(input_img)
                if self.target_transform:
                    target_img = self.target_transform(target_img)

                inp_tar = (input_img, target_img)
                self.cache.append(inp_tar)


    def _check_number_of_imgs(self):
        n_inp_imgs = len(os.listdir(self.input_path))
        n_out_imgs = len(os.listdir(self.target_path))
        if(n_inp_imgs != n_out_imgs):
            raise ValueError("The number of images does not match")
        self.dataset_len = n_inp_imgs

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        
        if self.use_cache:
            return self.cache[idx]

        input_img = read_image(os.path.join(self.input_path, self.filenames[idx]))
        target_img = read_image(os.path.join(self.target_path, self.filenames[idx]),
                                mode=ImageReadMode.GRAY)

        if self.transform:
            input_img = self.transform(input_img)
        if self.target_transform:
            target_img = self.target_transform(target_img)
        return input_img, target_img