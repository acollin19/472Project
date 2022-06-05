# Preprocessing, loading & analyzing the datasets

# Import PyTorch libraries
import matplotlib
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pandas as pd
from matplotlib import image as mp_image
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
from PIL import Image, ImageOps

from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import shutil
import os

img_folder = '../images'

# All images are 128x128 pixels
img_size = (250, 250)
num_workers = 2
batch_size = 20

# Sorted list of subfolders for each class
all_classes = sorted(os.listdir(img_folder))

# #convert to jpg
# for root, folders, files in os.walk(img_folder):
#     for directories in folders:
#         file_names = os.listdir(os.path.join(root, directories))
#         for file_name in file_names:
#             file_path = os.path.join(root, directories, file_name)
#             if os.path.isfile(file_path):
#                 image = Image.open(file_path)
#                 rgb_img = image.convert('RGB').save(file_path)


def resize_image(src_image, size=(250, 250), bg_color="white"):
    # resize the image so the longest dimension matches our target size
    src_image.thumbnail(size, Image.ANTIALIAS)

    # Create a new square background image
    new_image = Image.new("RGB", size, bg_color)

    # Paste the resized image into the center of the square background
    new_image.paste(src_image, (int((size[0] - src_image.size[0]) / 2), int((size[1] - src_image.size[1]) / 2)))

    return new_image


# create output folder if it doesn't exist
resized_img = '../resized_images'
if os.path.exists(resized_img):
    shutil.rmtree(resized_img)

for root, folders, files in os.walk(img_folder):
    for directories in folders:
        outputFolder = os.path.join(resized_img, directories)
        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)
        # Loop through the files in the directories
        file_names = os.listdir(os.path.join(root, directories))
        for file_name in file_names:
            # Open the file
            file_path = os.path.join(root, directories, file_name)
            image = Image.open(file_path)
            # Create a resized version and save it
            resized_image = resize_image(image, img_size)
            saveAs = os.path.join(outputFolder, file_name)
            resized_image.save(saveAs)


def pre_processing(data_path):
    # normalize data
    transformation = transforms.Compose([
        # Resize image
        # transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.3),
        transforms.ToTensor(),
        # Normalize the pixel values (in R, G, and B channels)
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load & transform all of the images
    full_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transformation
    )
    # # Split training/ testing (75% / 25%)
    # train_size = int(0.75 * len(full_dataset))
    # test_size = len(full_dataset) - train_size
    #
    # # use torch.utils.data.random_split for training/test split
    # train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    #
    # # loader for training
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=20,
    #     num_workers=2,
    #     shuffle=True
    # )
    #
    # # loader for testing
    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset,
    #     batch_size=20,
    #     num_workers=2,
    #     shuffle=False
    # )

    # return train_loader, test_loader
    return full_dataset


# train_loader, test_loader = pre_processing(img_folder)
dataset = pre_processing(resized_img)

# Split training/ testing (75% / 25%)
train_size = int(0.75 * len(dataset))
test_size = len(dataset) - train_size

# use torch.utils.data.random_split for training/test split
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# TRAINING LOADER
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=20,
    num_workers=2,
    shuffle=True
)

# TESTING LOADER
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=20,
    num_workers=2,
    shuffle=False
)
