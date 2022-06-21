import os
import shutil

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

img_folder = '../images_copy'

# All images are 64x64 pixels
img_size = (64, 64)


# Sorted list of subdirectories for each class
def get_classes(img_path):
    all_classes = sorted(os.listdir(img_path))
    if '.DS_Store' in all_classes:
        all_classes.remove('.DS_Store')
    print("ALL CLASSES ", all_classes)
    return all_classes


# CONVERT TO JPEG
def convert_to_jpeg():
    for root, folders, files in os.walk(img_folder):
        for directories in folders:
            file_names = os.listdir(os.path.join(root, directories))
            for file_name in file_names:
                file_path = os.path.join(root, directories, file_name)
                if os.path.isfile(file_path):
                    if not file_path.endswith(".jpeg"):
                        image = Image.open(file_path)
                        fn, fext = os.path.splitext(file_path)
                        conv_img = image.convert('RGB')
                        conv_img.save('{}.jpeg'.format(fn))
                        print("file_path ", file_path)
                        os.remove(file_path)


# RESIZE FUNCTION
def resize_image(src_image):
    basewidth = 64
    percent = (basewidth / float(src_image.size[0]))
    # height_size = int((float(src_image.size[1]) * float(percent)))
    height_size = 64
    new_image = src_image.resize((basewidth, height_size), Image.ANTIALIAS)
    return new_image


# CREATE FOLDER FOR RESIZED IMGS IF IT DOESN'T EXIST
def resize_save():
    # resized_img = '../resized_images_copy'
    # if os.path.exists(resized_img):
    #     shutil.rmtree(resized_img)

    # RESIZE AND SAVE IMAGES
    for root, folders, files in os.walk(img_folder):
        for directories in folders:
            # output_folder = os.path.join(resized_img, directories)
            output_folder = root + "/" + directories
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # Store image names in list
            file_names = os.listdir(os.path.join(root, directories))
            # Loop through all images
            for file_name in file_names:
                # Open the file in path
                file_path = os.path.join(root, directories, file_name)
                if os.path.isfile(file_path):
                    img = Image.open(file_path)
                    # Resize images and save in directory
                    rsz_img = resize_image(img)
                    save_img = os.path.join(output_folder, file_name)
                    rsz_img.save(save_img)


def pre_processing(data_path):
    # Normalize data
    transformation = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.3),
        # transforms.InterpolationMode.BICUBIC,
        # transforms.RandomRotation(20),
        # transforms.RandomResizedCrop(128),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # Normalize the pixel values (in R, G, and B channels)
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load & transform all of the images
    full_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transformation
    )
    # Split training/ testing (75% / 25%)
    train_size = int(0.75 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    # Use torch.utils.data.random_split for training/test split
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    # Loader for training
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        num_workers=2,
        shuffle=True
    )

    # Loader for testing
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1000,
        num_workers=2,
        shuffle=False
    )

    return (train_loader, test_loader), (train_dataset, test_dataset)


if __name__ == '__main__':
    resize_save()
