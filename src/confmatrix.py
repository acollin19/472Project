import torch
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix, f1_score
import numpy as np
from tabulate import tabulate
from skorch import NeuralNetClassifier
from torch.utils.data import random_split
import torch.optim as optim
import torch.nn as nn
import training
import preprocessing
from sklearn.metrics import confusion_matrix, classification_report

modelB = training.CNN()
modelB.load_state_dict(torch.load('saved_model'), strict=False)
_, (train_dataset, test_dataset) = preprocessing.pre_processing('../resized_images')

torch.manual_seed(0)
DEVICE = torch.device('cpu')

net = NeuralNetClassifier(
    # cnn=training.cnn(),
    # training.CNN,
    modelB,
    max_epochs=1,
    iterator_train__num_workers=4,
    iterator_valid__num_workers=4,
    lr=1e-3,
    batch_size=64,
    optimizer=optim.Adam,
    criterion=nn.CrossEntropyLoss,
    device=DEVICE
)


# data_len = len(train_dataset)
# split_train, val_data = random_split(train_dataset, [int(data_len - data_len * 0.2), int(data_len * 0.2)])

def evaluation():
    y_train = np.array([y for x, y in iter(train_dataset)])
    net.fit(train_dataset, y=y_train)
    y_pred = net.predict(test_dataset)
    y_true = np.array([y for x, y in iter(test_dataset)])
    # accuracy_score(y_true, y_pred)
    target_names = preprocessing.get_classes()
    class_rep = classification_report(y_true, y_pred, target_names=target_names)
    confusion_matrix(y_true, y_pred, labels=target_names)
    print("Classification Report ", class_rep)


if __name__ == '__main__':
    evaluation()
