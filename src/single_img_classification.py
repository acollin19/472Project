import os

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from src import training, preprocessing

model = training.CNN()
model.load_state_dict(torch.load('saved_model'), strict=False)

target_names = preprocessing.get_classes()
if '.DS_Store' in target_names:
    target_names.remove('.DS_Store')


def predict_image(data_path):
    print('Opening Image: {data_path}'.format(data_path=data_path))
    image = Image.open(data_path)
    x = TF.to_tensor(image)
    x.unsqueeze_(0)

    output = model(x)
    _, predicted = torch.max(output.data, 1)
    print('Predicted class: {predicted}'.format(predicted=target_names.__getitem__(predicted[0])))
    return predicted[0]


def test_all_images_in_directory(directory='../resized_images/ClothMask'):
    counter = [0, 0, 0, 0]
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if '.DS_Store' in f:
            continue
        # Check if it is a file
        if os.path.isfile(f):
            counter[predict_image(f)] += 1
    print(target_names)
    print(counter)


if __name__ == '__main__':
    predict_image('../resized_images/SurgicalMask/0003.jpeg')
    # test_all_images_in_directory('../resized_images/SurgicalMask')
