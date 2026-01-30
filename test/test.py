import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from lib.lib import fft_filtering, FFT

cap = cv2.VideoCapture(0)
if cap.isOpened():
    open, frame = cap.read()
else:
    open = False

cv2.namedWindow("radius")
cv2.createTrackbar("radius", "radius", 30, int(frame.shape[0]/2), lambda x:None)
cv2.createTrackbar("threshold", "radius", 127, 255, lambda x:None)

while open:
    ret, frame = cap.read()
    if not ret:
        break
    radius = cv2.getTrackbarPos("radius", "radius")
    threshold = cv2.getTrackbarPos("threshold", "radius")
    high_pass_filter = fft_filtering(frame, radius=radius)
    binary = cv2.threshold(high_pass_filter, threshold, 255, cv2.THRESH_BINARY)[1]
    kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernal)

    cv2.imshow("high_pass", high_pass_filter)
    cv2.imshow("binary", binary)
    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
    