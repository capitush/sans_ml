import time

import cv2
import torch
import numpy as np
import torchvision

import torch.nn.functional as F
import torch.nn as nn

bottomLeftCornerOfText = (90,90)
fontScale              = 10
fontColor              = (0,255,0)
lineType               = 2

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
NOPRESS = False
def press(keys):
    if NOPRESS:
        return
    for key in keys:
        if NOPRESS:
            return
        keyboard.press(key)

def release(keys):
    for key in keys:
        # print("released", key)
        keyboard.release(key)

classes = ['up', 'down', 'left', 'right', 'upleft', 'upright', 'downleft', 'downright', 'none']
current_key = []
cap = cv2.VideoCapture(0)
min_YCrCb = np.array([0, 133, 77], np.uint8)
max_YCrCb = np.array([235, 173, 127], np.uint8)
width, height = int(640/4), int(320/4)
net = Net(width, height)
PATH = "sans_model8.pth"
net.load_state_dict(torch.load(PATH))

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

last_click = time.time()
if NOPRESS:
    print("NOPRESS")

while(True):
    img = cv2.flip(cap.read()[1], 1)
    image_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    skinRegionYCrCb = cv2.inRange(image_YCrCb, min_YCrCb, max_YCrCb)
    view_img = cv2.bitwise_and(img, img, mask = skinRegionYCrCb)
    img = cv2.resize(view_img, (width, height))
    img = np.moveaxis(img, -1, 0)
    output = net(torch.tensor([img]))
    _, predicted = torch.max(output, 1)
    key = classes[predicted[0].item()]
    text_image = np.zeros((100, 640, 3))
    cv2.putText(text_image, key,
                (0,90),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                fontColor,
                lineType)
    h1, w1 = text_image.shape[:2]
    h2, w2 = view_img.shape[:2]
    vis = np.zeros((h1 + h2, max(w1,w2), 3), np.uint8)
    vis[:h1, :, :3] = text_image
    vis[h1:h1+h2, :, :3] = view_img
    cv2.imshow("A", vis)
    key = from_str_to_key(key)
    if key != current_key:
        if len(key) == 0:
            release(current_key)
        else:
            if current_key:
                release(current_key)
            press(key)
        current_key = key
    # good_contour = get_contour(skinRegionYCrCb)
    # area = cv2.contourArea(good_contour)
    # if area >= 21000 and time.time() - last_click > 0.5:
    #     # if not NOPRESS:
    #         # keyboard.press("z")
    #     last_click = time.time()

    if cv2.waitKey(2) & 0xFF == ord("w"):
        break
