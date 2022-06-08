import matplotlib.pyplot as plt
# from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix, f1_score
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
from skorch import NeuralNetClassifier

import preprocessing
import training

modelB = training.CNN()
modelB.load_state_dict(torch.load('saved_model'), strict=False)
_, (train_dataset, test_dataset) = preprocessing.pre_processing('../resized_images')
target_names = preprocessing.get_classes()

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
