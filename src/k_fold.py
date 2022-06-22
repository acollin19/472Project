import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay, make_scorer, \
    precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, ConcatDataset
import preprocessing
from cnn import CNN

all_imgs = '../images_copy'
female_imgs = '../images_copy/Female'
male_imgs = '../images_copy/Male'
old_imgs = '../images_copy/Old'
young_imgs = '../images_copy/Young'

_, (train_dataset, test_dataset) = preprocessing.pre_processing(all_imgs)
dataset = ConcatDataset([train_dataset, test_dataset])

torch.manual_seed(0)

modelB = CNN()
optimizer = torch.optim.Adam(modelB.parameters(), lr=1e-3)
loss_function = nn.CrossEntropyLoss()
num_epochs = 1
num_folds = 2
results = {}
eval_res = []


def k_fold_new():
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=10)
    # for fold, (train_ids, test_ids) in enumerate(kfold.split(train_dataset)):
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        trainloader = torch.utils.data.DataLoader(dataset, batch_size=32, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(dataset, batch_size=32, sampler=test_subsampler)

        # train
        for epoch in range(0, num_epochs):
            current_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, targets = data
                optimizer.zero_grad()
                outputs = modelB(inputs)
                loss = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()

                # statistics
                current_loss += loss.item()
                if i % 500 == 499:
                    print('Loss after mini-batch %5d: %.3f' %
                          (i + 1, current_loss / 500))
                    current_loss = 0.0

        print('Starting testing')

        correct, total = 0, 0
        with torch.no_grad():

            # Iterate over the test data and generate predictions
            for i, data in enumerate(testloader, 0):
                # Get inputs
                inputs, target = data

                # Generate outputs
                outputs = modelB(inputs)

                # Set total and correct
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

            precision = precision_score(target, predicted, average="weighted", zero_division=1)
            recall = recall_score(target, predicted, average="weighted", zero_division=1)
            f1 = f1_score(target, predicted, average="weighted", zero_division=1)
            accuracy = accuracy_score(target, predicted)
            eval_res.append([precision, recall, f1, accuracy])
            results[fold] = 100.0 * (correct / total)

            print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
            print('Precision for fold %d: %d %%' % (fold, precision * 100))
            print('Recall for fold %d: %d %%' % (fold, 100.0 * recall))
            print('Other Accuracy for fold %d: %d %%' % (fold, 100.0 * accuracy))
            print('F1 for fold %d: %d %%' % (fold, 100.0 * f1))

        # Print fold results
        print(f'K-FOLD CROSS VALIDATION RESULTS FOR {num_folds} FOLDS')
        print('--------------------------------')
        res_sum = 0.0
        for key, value in results.items():
            print(f'Fold {key}: {value} %')
            res_sum += value
        print(f'Average Accuracy: {res_sum / len(results.items())} %')

        # Saving the model
        save_path = f'./model-K-fold-{fold}.pth'
        torch.save(modelB.state_dict(), save_path)
    return eval_res


if __name__ == '__main__':
    k_fold_new()
