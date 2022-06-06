import sys
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td

import preprocessing


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=320, kernel_size=3, padding=1),
            nn.BatchNorm2d(320),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=320, out_channels=320, kernel_size=3, padding=1),
            nn.BatchNorm2d(320),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=320, out_channels=640, kernel_size=3, padding=1),
            nn.BatchNorm2d(640),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=640, out_channels=640, kernel_size=3, padding=1),
            nn.BatchNorm2d(640),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(8 * 8 * 640, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=80*80*640, out_features=4)
        )

    def forward(self, x):
        # Conv layers
        x = self.conv_layer(x)  # Flatten
        x = x.view(x.size(0), -1)

        # fc layer
        x = self.fc_layer(x)
        return x


def cnn():
    # Parameters that can be tuned
    num_epochs = 4
    num_classes = 4
    learning_rate = 0.001
    train_loader, test_loader = preprocessing.pre_processing('../resized_images')
    classes = ('no_mask', 'cloth_mask', 'surgical_mask', 'n95_mask')

    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())

            # Backprop and optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Train accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

            if (i + 1) % 100 == 0:
                print(
                    'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch + 1, num_epochs, i + 1,
                                                                                          total_step, loss.item(),
                                                                                          (correct / total) * 100))
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))


if __name__ == '__main__':
    v = sys.version_info
    print('Python version: {v0}.{v1}.{v2}'.format(v0=v[0], v1=v[1], v2=v[2]))
    cnn()
