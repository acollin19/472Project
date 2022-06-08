import torch
# from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix, f1_score
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from skorch import NeuralNetClassifier
from torch.utils.data import random_split
import torch.optim as optim
import torch.nn as nn
import training
import preprocessing
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix, accuracy_score, ConfusionMatrixDisplay

modelB = training.CNN()
modelB.load_state_dict(torch.load('saved_model'), strict=False)
_, (train_dataset, test_dataset) = preprocessing.pre_processing('../resized_images')
target_names = preprocessing.get_classes()

torch.manual_seed(0)
DEVICE = torch.device('cpu')

net = NeuralNetClassifier(
    modelB,
    max_epochs=1,
    iterator_train__num_workers=0,
    iterator_valid__num_workers=0,
    lr=1e-3,
    batch_size=128,
    optimizer=optim.Adam,
    criterion=nn.CrossEntropyLoss,
    device=DEVICE
)


def evaluation():
    y_train = np.array([y for x, y in iter(train_dataset)])
    net.fit(train_dataset, y=y_train)
    y_pred = net.predict(test_dataset)
    y_true = np.array([y for x, y in iter(test_dataset)])
    accuracy_score(y_true, y_pred)
    print("Class", classification_report(y_true, y_pred, target_names=target_names))
    #plot_confusion_matrix(net, test_dataset, y_true)
    ConfusionMatrixDisplay.from_estimator(net, test_dataset, y_true.reshape(-1,1), display_labels=target_names)
    plt.show()


if __name__ == '__main__':
    evaluation()
