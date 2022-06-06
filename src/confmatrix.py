
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix, f1_score
import numpy as np
from tabulate import tabulate
from skorch import NeuralNetClassifier
import torch.optim as optim
import torch.nn as nn

# Already done in prev question
net = NeuralNetClassifier(
       cnn,
       max_epochs=1,
       iterator_train__num_workers=4,
       iterator_valid__num_workers=4,
       lr=1e-3,
       batch_size=64,
       optimizer=optim.Adam,
       criterion=nn.CrossEntropyLoss,
)
# dataset and train_dataset from preprocessing.py
y_train = np.array([y for x, y in iter(train_dataset)])
net.fit(train_dataset, y=y_train)

y_true = np.array([y for x, y in iter(dataset)])
y_pred = net.predict(dataset)


# calculate accuracy, precision, f1-measure, recall given confusion matrix
def confusion_matrix(true, predicted):
    cm = confusion_matrix(true, predicted)
    accuracy = accuracy_score(true, predicted)
    f1 = f1_score(true, predicted)
    precision = precision_score(true, predicted)
    recall = recall_score(true, predicted)

    # compiling results into a table
    table = [['Accuracy', accuracy],
             ['Precision', precision],
             ['Recall', recall],
             ['F1-measure', f1],
             ['Confusion Matrix', cm]]
    results = tabulate(table, tablefmt='fancy_grid')

    return results


confusion_matrix(y_true, y_pred)
