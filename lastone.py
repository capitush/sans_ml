import cv2
import torch
import numpy as np
import torchvision
import time

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


import pyautogui


classes = ['up', 'down', 'left', 'right', 'none', 'upright', 'upleft', 'downright', 'downleft']
current_key = []
cap = cv2.VideoCapture(0)
min_YCrCb = np.array([0, 133, 77], np.uint8)
max_YCrCb = np.array([235, 173, 127], np.uint8)
width, height = int(640/8),int(480/8)
net = Net(width, height)
PATH = "sans_model7c.pth"
net.load_state_dict(torch.load(PATH))
import math
last_click = time.time()

def get_area(cnt):
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    return cv2.contourArea(cnt) if cv2.contourArea(cnt) > 300 else 0


def get_center_contour(cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    return int(x + w / 2), int(y + h / 2)


def get_contour(frame):

    # Find the contours in the frame
    contours, hierarchy = cv2.findContours(frame, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour in the red rectangle:

    # Sort them by area:
    contours.sort(key=get_area, reverse=True)
    # Draw the red rectangle:
    try:
        return contours[0]
    except:
        return None


while(True):
    frame = cv2.flip(cap.read()[1], 1)
    image_YCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    skinRegionYCrCb = cv2.inRange(image_YCrCb, min_YCrCb, max_YCrCb)
    img = cv2.bitwise_and(frame, frame, mask = skinRegionYCrCb)
    cv2.imshow("A", img)
    img = cv2.resize(img, (width, height))
    img = np.moveaxis(img, -1, 0)
    output = net(torch.tensor([img]))
    _, predicted = torch.max(output, 1)
    key = classes[predicted[0].item()]
    print(key)
    dx = 0
    dy = 0
    if "up" in key:
        dy = -1
    if "down" in key:
        dy = 1
    if "left" in key:
        dx = -1
    if "right" in key:
        dx = 1
    scalar = 10
    pyautogui.move(scalar*dx, scalar*dy)
    good_contour = get_contour(skinRegionYCrCb)
    area = cv2.contourArea(good_contour)
    if area >= 21000 and time.time() - last_click > 0.5:
        pyautogui.click()
        last_click = time.time()
    cv2.drawContours(frame, [good_contour], -1, (0, 0, 255), -1)
    cv2.imshow("b", frame)
    if cv2.waitKey(2) & 0xFF == ord("w"):
        break