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
import training

modelB = training.CNN()
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

# target_names = preprocessing.get_classes()
# if '.DS_Store' in target_names:
#     target_names.remove('.DS_Store')

torch.manual_seed(0)
device = torch.device('cpu')
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


def evaluation_Female():
    target_names = preprocessing.get_classes(female_imgs)
    if '.DS_Store' in target_names:
        target_names.remove('.DS_Store')

    y_train = np.array([y for x, y in iter(train_datasetF)])
    net.fit(train_datasetF, y=y_train)
    y_pred = net.predict(test_datasetF)
    y_true = np.array([y for x, y in iter(test_datasetF)])
    accuracy_score(y_true, y_pred)
    print("Class", classification_report(y_true, y_pred, target_names=target_names))
    # plot_confusion_matrix(net, test_dataset, y_true)
    ConfusionMatrixDisplay.from_estimator(net, test_datasetF, y_true.reshape(-1, 1), display_labels=target_names)
    plt.show()


def evaluation_Male():
    target_names = preprocessing.get_classes(male_imgs)
    if '.DS_Store' in target_names:
        target_names.remove('.DS_Store')

    y_train = np.array([y for x, y in iter(train_datasetM)])
    net.fit(train_datasetM, y=y_train)
    y_pred = net.predict(test_datasetM)
    y_true = np.array([y for x, y in iter(test_datasetM)])
    accuracy_score(y_true, y_pred)
    print("Class", classification_report(y_true, y_pred, target_names=target_names))
    # plot_confusion_matrix(net, test_dataset, y_true)
    ConfusionMatrixDisplay.from_estimator(net, test_datasetM, y_true.reshape(-1, 1), display_labels=target_names)
    plt.show()


def evaluation_Old():
    target_names = preprocessing.get_classes(old_imgs)
    if '.DS_Store' in target_names:
        target_names.remove('.DS_Store')

    y_train = np.array([y for x, y in iter(train_datasetO)])
    net.fit(train_datasetO, y=y_train)
    y_pred = net.predict(test_datasetO)
    y_true = np.array([y for x, y in iter(test_datasetO)])
    accuracy_score(y_true, y_pred)
    print("Class", classification_report(y_true, y_pred, target_names=target_names))
    # plot_confusion_matrix(net, test_dataset, y_true)
    ConfusionMatrixDisplay.from_estimator(net, test_datasetO, y_true.reshape(-1, 1), display_labels=target_names)
    plt.show()


def evaluation_Young():
    target_names = preprocessing.get_classes(young_imgs)
    if '.DS_Store' in target_names:
        target_names.remove('.DS_Store')

    y_train = np.array([y for x, y in iter(train_datasetY)])
    net.fit(train_datasetY, y=y_train)
    y_pred = net.predict(test_datasetY)
    y_true = np.array([y for x, y in iter(test_datasetY)])
    accuracy_score(y_true, y_pred)
    print("Class", classification_report(y_true, y_pred, target_names=target_names))
    # plot_confusion_matrix(net, test_dataset, y_true)
    ConfusionMatrixDisplay.from_estimator(net, test_datasetY, y_true.reshape(-1, 1), display_labels=target_names)
    plt.show()


def k_fold():
    k_train = np.array([y for x, y in iter(train_dataset)])
    net.fit(train_dataset, y=k_train)
    # k-fold
    train_sliceable = SliceDataset(train_dataset)
    scores = cross_val_score(net, train_sliceable, k_train, cv=10, scoring="accuracy")
    print("scores ", scores)


if __name__ == '__main__':
    k_fold()
    evaluation_Female()
    evaluation_Male()
    evaluation_Old()
    evaluation_Young()

