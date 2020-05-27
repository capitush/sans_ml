import cv2
import torch
import numpy as np
import torchvision

import torch.nn.functional as F
import torch.nn as nn

class Net(nn.Module):
    channels = 3
    height = 480
    width = 640
    num_classes = 9

    def __init__(self, width, height):
        super(Net, self).__init__()
        self.width = width
        self.height = height
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(in_features=self.width * self.height, out_features=120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x.float())))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.width * self.height)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


from pynput.keyboard import Key, Controller


def from_str_to_key(key):
    keys = []
    if "up" in key:
        keys.append(Key.up)
    if "down" in key:
        keys.append(Key.down)
    if "left" in key:
        keys.append(Key.left)
    if "right" in key:
        keys.append(Key.right)
    return keys

keyboard = Controller()

def press(keys):
    for key in keys:
        # print("pressed", key)
        keyboard.press(key)

def release(keys):
    for key in keys:
        # print("released", key)
        keyboard.release(key)

classes = ['up', 'down', 'left', 'right', 'none', 'upright', 'upleft', 'downright', 'downleft']
current_key = []
cap = cv2.VideoCapture(0)
min_YCrCb = np.array([0, 133, 77], np.uint8)
max_YCrCb = np.array([235, 173, 127], np.uint8)
width, height = int(640/8),int(480/8)
net = Net(width, height)
PATH = "sans_model7c.pth"
net.load_state_dict(torch.load(PATH))
while(True):
    img = cv2.flip(cap.read()[1], 1)
    image_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    skinRegionYCrCb = cv2.inRange(image_YCrCb, min_YCrCb, max_YCrCb)
    img = cv2.bitwise_and(img, img, mask = skinRegionYCrCb)
    cv2.imshow("A", img)
    img = cv2.resize(img, (width, height))
    img = np.moveaxis(img, -1, 0)
    output = net(torch.tensor([img]))
    _, predicted = torch.max(output, 1)
    key = classes[predicted[0].item()]
    print(key)
    key = from_str_to_key(key)
    # print(key)
    if key != current_key:
        if len(key) == 0:
            release(current_key)
        else:
            if current_key:
                release(current_key)
            press(key)
        current_key = key

    if cv2.waitKey(2) & 0xFF == ord("w"):
        break
