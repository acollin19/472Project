import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
from skorch import NeuralNetClassifier
from sklearn.model_selection import KFold, cross_val_score
from skorch.helper import SliceDataset

import preprocessing
from cnn import CNN

modelB = CNN()
modelB.load_state_dict(torch.load('saved_model'), strict=False)

all_imgs = '../images_copy'
female_imgs = '../images_copy/Female'
male_imgs = '../images_copy/Male'
old_imgs = '../images_copy/Old'
young_imgs = '../images_copy/Young'

# _, (train_dataset, test_dataset) = preprocessing.pre_processing('../resized_images')
_, (train_dataset, test_dataset) = preprocessing.pre_processing(all_imgs)
_, (train_datasetF, test_datasetF) = preprocessing.pre_processing(female_imgs)
_, (train_datasetM, test_datasetM) = preprocessing.pre_processing(male_imgs)
_, (train_datasetO, test_datasetO) = preprocessing.pre_processing(old_imgs)
_, (train_datasetY, test_datasetY) = preprocessing.pre_processing(young_imgs)

torch.manual_seed(0)

"""
In order to use mps on M1 Macs the following if statement needs to be added in the file skorch/utils.py around
line 140 after the if X.is_cuda: X = X.cpu() statement
if X.is_mps:
    X = X.cpu()
Also a nightly version of pytorch is required
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # For windows (will use cpu on macs)
# device = torch.device('mps' if torch.has_mps else 'cpu')  # For mac (M1 macs with nightly version of pytorch)
print("Device used to compute the confusion matrix: {device}".format(device=device))

net = NeuralNetClassifier(
    modelB,
    max_epochs=1,
    iterator_train__num_workers=0,
    iterator_valid__num_workers=0,
    lr=1e-3,
    batch_size=128,
    optimizer=optim.Adam,
    criterion=nn.CrossEntropyLoss,
    device=device
)


def evaluation(img_path, train_data, test_data):
    target_names = preprocessing.get_classes(img_path)
    y_train = np.array([y for x, y in iter(train_data)])
    net.fit(train_data, y=y_train)
    y_pred = net.predict(test_data)
    y_true = np.array([y for x, y in iter(test_data)])
    accuracy_score(y_true, y_pred)
    print("Class", classification_report(y_true, y_pred, target_names=target_names))
    # plot_confusion_matrix(net, test_dataset, y_true)
    ConfusionMatrixDisplay.from_estimator(net, test_data, y_true.reshape(-1, 1), display_labels=target_names)
    plt.show()


def k_fold():
    k_train = np.array([y for x, y in iter(train_dataset)])
    net.fit(train_dataset, y=k_train)
    # k-fold
    train_sliceable = SliceDataset(train_dataset)
    scores = cross_val_score(net, train_sliceable, k_train, cv=10, scoring="accuracy")
    print("scores ", scores)


if __name__ == '__main__':
    evaluation(female_imgs, train_datasetF, test_datasetF)
    evaluation(male_imgs, train_datasetM, test_datasetM)
    evaluation(old_imgs, train_datasetO, test_datasetO)
    evaluation(young_imgs, train_datasetY, test_datasetY)
    k_fold()
