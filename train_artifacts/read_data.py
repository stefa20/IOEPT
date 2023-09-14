# encoding: utf-8
"""
Read images and corresponding labels.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import cv2
import numpy as np
import pandas as pd
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def load_dataset_from_folder(data_path, transform, batch_size):
    """Loads data from folder of type folder/clases/images.
            data_path: path to the super folder containing folders with class names
            transforms: trnasforms.Compose object containing on-fly image transformation
                        and data augmentation.

            Returns: dataloader to iterate with enumarete(data_loader) or get one batch
                    with next(iter(data_loader))"""

    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    return train_loader

def process_image_file(filepath, back= 'PIL'):

    if back == 'cv':
        img = cv2.imread(filepath)
        # img = cv2.resize(img, (size, size))

    elif back == 'PIL':
        img = Image.open(filepath).convert('RGB')
    return img

def _process_csv_file(file):
    with open(file, 'r') as fr:
        files = fr.readlines()
    return files

def _process_excel_file(xlsx_path):
    data_frame = pd.read_excel(xlsx_path, sheet_name='Total', index_col=False)
    return data_frame

def get_all_values(d):
    if isinstance(d, dict):
        for v in d.values():
            yield from get_all_values(v)
    elif isinstance(d, list):
        for v in d:
            yield from get_all_values(v)
    else:
        yield d

def plotImages(batch_data, labels=None, title=None, n_images=(4, 4), gray=True ):
    fig, axes = plt.subplots(n_images[0], n_images[1], figsize=(12, 12))
    axes = axes.flatten()
    images = batch_data[0]
    plt.title(title)
    for n, ax in zip(range(n_images[0]*n_images[1]), axes):
        img = images[n].numpy()
        img = np.moveaxis(img, 0, -1)
        if gray:
            ax.imshow(img, cmap='gray') #, vmin=-1, vmax=1)
        else: ax.imshow(img) #, vmin=0, vmax=1) #, vmin=-3, vmax=3)#cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_xticks(())
        ax.set_yticks(())
        ind = int(np.argmax(batch_data[1][n], axis=-1))
        ind = int(batch_data[1][n])
        ax.set_title((labels[ind]))
    plt.tight_layout()
    plt.show()

def plotTensorList(tensor_list):
    n_images = len(tensor_list)
    fig, axes = plt.subplots(int(np.sqrt(n_images)), int(np.sqrt(n_images)), figsize=(12, 12))
    axes = axes.ravel()

    for i, tensor in zip(range(len(axes)), tensor_list):
        img = tensor.numpy()[0,...]
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(i)

    plt.tight_layout()
    plt.show()

class FER2013(Dataset):

    def __init__(self, pathImageDirectory, pathDatasetFile, transform=None, balance=None):

        self.listImagePaths = []
        self.listLabelVotes = []
        self.listImageLabels = []
        self.transform = transform

        # ---- Open file, get image paths and labels

        fileDescriptor = open(pathDatasetFile, "r")

        # ---- get into the loop
        line = True
        while line:

            line = fileDescriptor.readline()
            # --- if not empty
            if line:
                lineItems = line.split(',')

                imagePath = os.path.join(pathImageDirectory, lineItems[0])
                # imageVotes = lineItems[1:]
                # imageVotes = [int(i) for i in lineItems[1:]]
                # imageLabel = [label/max(imageVotes) > 0.75 for label in imageVotes]
                # imageLabel = np.array(imageLabel, dtype=int)

                imageLabel = int(lineItems[1])

                self.listImagePaths.append(imagePath)
                # self.listLabelVotes.append(imageVotes)
                self.listImageLabels.append(imageLabel)

        if balance is not None:
            balanced_imgs, balanced_labels = self.balance(self.listImagePaths, self.listImageLabels, balance)
            self.listImagePaths = balanced_imgs
            self.listImageLabels = balanced_labels

        fileDescriptor.close()

        u, c = np.unique(self.listImageLabels, return_counts=True)
        print('Label distribution', dict(zip(u, c)))

        # self.imageClass = np.argmax(self.listImageLabels, axis=-1)

    def __getitem__(self, index):

        imagePath = self.listImagePaths[index]

        imageData = Image.open(imagePath).convert('L')
        # imageLabel = torch.FloatTensor(self.listImageLabels[index])
        imageLabel = torch.tensor(self.listImageLabels[index])
        if self.transform != None: imageData = self.transform(imageData)

        return imageData, imageLabel

    def __len__(self):

        return len(self.listImagePaths)

    def balance(self, files, targets, num):
        """ Apply over and under sample techniques to balance the targets of a data set
            args:
            files: iterable of dataset samples
            targets: iterable of dataset targets
            num: amount of observations per class after balance."""

        files, targets = self.over_sample(files, targets, num)
        files, targets = self.under_sample(files, targets, num)

        return files, targets

    def under_sample(self, files, targets, num):
        unique, counts = np.unique(targets, return_counts=True)
        # emotions_id = targets  # np.argmax(emotions, axis=1)

        index_list = list()
        for i in unique:
            index = np.where(targets == i)[0]
            if np.shape(index)[0] > num:
                if i == 6:  ### special balance for any class
                    index_list.append(index[num:])
                else:
                    index_list.append(index[num:])

        index_4_delete = np.concatenate(index_list)

        balanced_faces = np.delete(files, index_4_delete, axis=0)
        balanced_emotions = np.delete(targets, index_4_delete, axis=0)

        # balanced_emotions = np.where(balanced_emotions == 4, 3, balanced_emotions)
        # balanced_emotions = np.where(balanced_emotions == 6, 4, balanced_emotions)

        unique, counts = np.unique(balanced_emotions, return_counts=True)
        return balanced_faces, balanced_emotions

    def over_sample(self, files, targets, num):
        unique, counts = np.unique(targets, return_counts=True)

        index_list = list()
        for (i, count) in zip(unique, counts):
            while count < num:
                index = np.where(targets == i)[0]
                res_files = [files[i] for i in index]
                res_targets = [targets[i] for i in index]
                files = files + res_files
                targets = targets + res_targets
                count = len(np.where(targets == i)[0])

        return files, targets

def get_transforms(augment=False):

    if augment:

        transformations = transforms.Compose([
            # transforms.ToPILImage(mode='RGB'),
            transforms.ColorJitter(brightness=(0.8,1.2),contrast=(0.8,1.2), saturation=(0.8,1.2)),
            transforms.RandomResizedCrop(48, scale=(0.7, 1.2)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor()
        ])
    else:

        transformations = transforms.Compose([
            transforms.ToTensor()
        ])

    return transformations

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    path = '/media/lecun/HD/Grimmat/Emotions Video/Fer2012/FER2013/FER2013'
    processing = get_transforms(augment=True)
    data = FER2013(f'{path}Train/', f'{path}Train/labels.csv', transform=processing, balance=3000)
    u,c = np.unique(data.listImageLabels, return_counts=True)
    # print(dict(zip(u, c)))
    labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    plt.bar(u,c, tick_label=labels)
    plt.show()

    loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)
    for _ in range(10):
        batch = next(iter(loader))
        plotImages(batch, labels=labels, n_images=(4,8))
        plt.show()