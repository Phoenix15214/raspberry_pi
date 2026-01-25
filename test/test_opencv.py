import cv2
import numpy as np


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

cv2.namedWindow("Camera Test")
cv2.createTrackbar("Threshold", "Camera Test", 120, 255, lambda x: None)
threshold_value = 120
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头帧")
            break
        
        cv2.imshow('Camera Test', frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        threshold_value = cv2.getTrackbarPos("Threshold", "Camera Test")
        binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)[1]
        cv2.imshow('Oringinal_binary', binary)
        # kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        cv2.imshow('Binary', binary)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()