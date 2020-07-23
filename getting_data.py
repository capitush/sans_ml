import cv2
import numpy as np
import time
cap = cv2.VideoCapture(0)
min_YCrCb = np.array([0, 133, 77], np.uint8)
max_YCrCb = np.array([235, 173, 127], np.uint8)
direction = "upleft"
i = 0
time.sleep(1)
while True:

    img = cv2.flip(cap.read()[1], 1)
    image_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    skinRegionYCrCb = cv2.inRange(image_YCrCb, min_YCrCb, max_YCrCb)
    frame = cv2.bitwise_and(img, img, mask = skinRegionYCrCb)
    name = "images8\\{}.png".format(direction + str(i))
    i += 1

    cv2.imshow("A", frame)
    cv2.imwrite(name, frame)

    if cv2.waitKey(2) & 0xFF == ord("w"):
        break