import torchvision.transforms as transforms
from torchvision.datasets import VisionDataset
from torch.utils.data import Subset
from functools import reduce
from sklearn.model_selection import StratifiedShuffleSplit
from PIL import Image
import os
import glob
import torch
import pandas as pd
import numpy as np


def pil_loader(path, grayscale=False):
    with open(path, 'rb') as f:
        img = Image.open(f)
        target = 'L' if grayscale else 'RGB'
        return img.convert(target)


def load_labels(csv_path, labels_columns, labels_ids):
    
    labels_ids = labels_ids.values() if isinstance(labels_ids, dict) else labels_ids

    tsv = pd.read_csv(csv_path, sep='\t')
    labels = reduce(lambda a,b: a + b, [tsv[labels_column] for labels_column in labels_columns])
    labels = pd.Series(pd.Categorical(labels, categories=labels_ids))

    labels_values = torch.tensor(labels.cat.codes)
    labels_ids = dict(zip(range(len(labels.cat.categories)), labels.cat.categories))

    return labels_values, labels_ids, tsv["file_name"].apply(lambda x: ".".join(x.split(".")[:-1])) 

class ImageDataset():
    def __init__(self, folder, train_ram=False, test_ram=False, **kwargs):
        
        if "all_in_ram" not in kwargs:
            kwargs["all_in_ram"] = False
        
        if kwargs["all_in_ram"] == True:
            self.train_ram = True
            self.test_ram = True
        else:
            self.train_ram = train_ram
            self.test_ram = test_ram
        
        kwargs["all_in_ram"] = self.train_ram
        self.train_dataset = ImageDatasetSplit(folder, split="train", **kwargs)

        kwargs["all_in_ram"] = self.test_ram
        self.test_dataset = ImageDatasetSplit(folder, split="test", **kwargs)

    def train_val_test_split(self):
        return self.train_dataset, self.test_dataset

    def standardize(self):
        pass


class ImageDatasetSplit(VisionDataset):
    def __init__(self, folder, split, transform=None, target_transform=None, all_in_ram=False, labels_columns=('sample_type', 'project_id'), labels_ids=None, grayscale=False):
        super().__init__(folder, transform=transform, target_transform=target_transform)
            
        self.data = []
        self.transform = transform
        self.all_in_ram = all_in_ram
        self.grayscale = grayscale
        self.split = split

        paths = open(os.path.join(folder, split + '_images.txt'), 'r')

        labels, self.labels_ids, filenames = load_labels(os.path.join(folder, "images_labels.txt"), labels_columns, labels_ids)
        labels_dict = {im:label for (im,label) in zip(filenames,labels)}
        self.labels = []

        if self.all_in_ram:
            for img in paths:
                image = pil_loader(os.path.join(folder, img.rstrip()), grayscale)
                self.data.append(image)
                self.labels.append(labels_dict[img.split("/")[-2]])
        else:
            for img in paths:
                self.data.append(os.path.join(folder, img.rstrip()))
                self.labels.append(labels_dict[img.split("/")[-2]])
        self.labels = torch.stack(self.labels)    
     
    def __getitem__(self, index):
        image = self.data[index]
        label = self.labels[index]
        if not self.all_in_ram:
            image = pil_loader(image, self.grayscale)

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        return image, label
    
    def __len__(self):
        return len(self.data)
