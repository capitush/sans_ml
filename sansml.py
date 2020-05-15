import os
from PIL import Image
import numpy as np
import torch
import cv2
import torchvision

import torch.nn.functional as F
import torch.nn as nn


class Net(nn.Module):
    channels = 3
    height = 480
    width = 640

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x.float())))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


classes = ['up', 'down', 'left', 'right', 'none']


class DataSet(object):
    def __init__(self, transforms):
        self.root = ""
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(self.root, "images"))))

    def __getitem__(self, idx):
        # load images
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        img = cv2.imread(img_path)
        img = np.moveaxis(img, -1, 0)

        img_label = img_path.split("\\")[1]
        img_label = img_label.split(".")[0]
        img_label = ''.join([i for i in img_label if not i.isdigit()])
        img_label = classes.index(img_label)
        # print("Label: " + img_label + " Index: " + str(idx))
        # there is only one class

        if self.transforms is not None:
            img, target = self.transforms(img, img_label)

        return img, img_label

    def __len__(self):
        return len(self.imgs)


import matplotlib.pyplot as plt


import torchvision.transforms as transforms
import torch.optim as optim


def main():
    # get some random training images
    transform = transforms.Compose([transforms.ToTensor()]) # Defing PyTorch Transform


    trainset = DataSet(None)
    testset = DataSet(None)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    net = Net()
    VERY_NICE_criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = VERY_NICE_criterion(outputs, torch.tensor(labels))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss))
                running_loss = 0.0

    print('Finished Training')
    PATH = "sans_model.pth"
    torch.save(net.state_dict(), PATH)

def AAAAAAA():
    root = "C:\\Users\\1\PycharmProjects\SERVER\ML\sans_ml\images\\"
    imgs = list(os.listdir("images"))
    i = 0
    for image in imgs:
        img = cv2.imread(root + image)
        img = cv2.resize(img, (32, 32))
        cv2.imwrite(root + image, img)
        print(i * 100 / len(imgs))
        i += 1

def test():
    data = DataSet(None)
    # trainloader = iter(torch.utils.data.DataLoader(DataSet(None), batch_size=4,
    #                                           shuffle=True, num_workers=2))
    image, label = data[300]
    print(label)
    net = Net()
    PATH = "sans_model.pth"
    net.load_state_dict(torch.load(PATH))
    output = net(torch.tensor([image]))
    _, predicted = torch.max(output, 1)
    print(predicted)


if __name__ == '__main__':
    # main()
    # AAAAAAA()
    test()