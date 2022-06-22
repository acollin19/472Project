import torch
import torch.nn as nn

import preprocessing
from src.cnn import CNN


def training():
    # Parameters that can be tuned
    num_epochs = 50
    learning_rate = 0.001
    # loaders, _ = preprocessing.pre_processing('../new_images_all')
    loaders, _ = preprocessing.pre_processing('../resized_images_init')
    train_loader, test_loader = loaders

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # For Windows (will use cpu on macs)
    device = torch.device('mps' if torch.has_mps else 'cpu')  # For Mac (M1 Macs only with nightly version of pytorch)
    print("Device used to train the model: {device}".format(device=device))

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
        print('epoch {e}'.format(e=epoch))
        for i, data in enumerate(train_loader):
            images, labels = data[0].to(device), data[1].to(device)
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

            if (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), (correct / total) * 100))

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the {total} test images: {results} %'
              .format(total=total, results=(correct / total) * 100))

    torch.save(model.state_dict(), 'saved_model')


if __name__ == '__main__':
    training()
