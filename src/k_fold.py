import torch
import torch.nn as nn
from sklearn.model_selection import KFold
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


def k_fold_new():
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=10)
    # for fold, (train_ids, test_ids) in enumerate(kfold.split(train_dataset)):
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        print(f'FOLD {fold}')

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

        # Saving the model
        save_path = f'./model-fold-{fold}.pth'
        torch.save(modelB.state_dict(), save_path)

        correct, total = 0, 0
        with torch.no_grad():

            # Iterate over the test data and generate predictions
            for i, data in enumerate(testloader, 0):
                # Get inputs
                inputs, targets = data

                # Generate outputs
                outputs = modelB(inputs)

                # Set total and correct
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

            # Print accuracy
            print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
            print('--------------------------------')
            results[fold] = 100.0 * (correct / total)

        # Print fold results
        print(f'K-FOLD CROSS VALIDATION RESULTS FOR {num_folds} FOLDS')
        print('--------------------------------')
        res_sum = 0.0
        for key, value in results.items():
            print(f'Fold {key}: {value} %')
            res_sum += value
        print(f'Average: {res_sum / len(results.items())} %')


if __name__ == '__main__':
    k_fold_new()
