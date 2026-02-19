import cv2
import numpy as np


cap = cv2.VideoCapture(0)
if cap.isOpened():
    open, read = cap.read()
else:
    open = False
cv2.namedWindow("Contours")
cv2.createTrackbar("Threshold", "Contours", 160, 255, lambda x:None)
MIN_AREA = 20000
while open:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    threshold = cv2.getTrackbarPos("Threshold", "Contours")
    binary = cv2.threshold(clahe, threshold, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    draw_img = frame.copy()
    # draw_img = cv2.drawContours(draw_img, contours, -1, (0, 255, 0), 2)
    for cnt in contours:
        epsilon = 0.1 * cv2.arcLength(cnt, False)
        approx = cv2.approxPolyDP(cnt, epsilon, False)
        contour_area = cv2.contourArea(cnt)
        contour_num = len(approx)
        if contour_area >= MIN_AREA and contour_num == 5:
            rotating_box = cv2.minAreaRect(approx)
            box = cv2.boxPoints(rotating_box)
            box = np.int32(box)
            draw_img = cv2.drawContours(draw_img, [box], -1, (0, 255, 0), 2)
    cv2.imshow('Contours', draw_img)
    cv2.imshow('Binary', binary)
    if cv2.waitKey(10) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()