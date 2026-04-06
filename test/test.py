import cv2
import numpy as np
import process_lib.image_lib as lb
import process_lib.control_lib as ctrl

cap = cv2.VideoCapture(0)

img = []

for i in range(8):
    path = rf"D:\zpb\2026\HUST_STI\Med_Car_Program\Template\{i+1}.jpg"
    img.append(cv2.imread(path))
    img[i] = cv2.cvtColor(img[i], cv2.COLOR_BGR2GRAY)
    img[i] = cv2.resize(img[i], (200, 200), interpolation=cv2.INTER_NEAREST)

if cap.isOpened():
    open, read = cap.read()
else:
    open = False

cv2.namedWindow("Contours")
cv2.namedWindow("Warped")
cv2.createTrackbar("Threshold", "Contours", 160, 255, lambda x:None)
MIN_AREA = 20000
warped = img[7]

while open:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8)).apply(gray)
    threshold = cv2.getTrackbarPos("Threshold", "Contours")
    binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)[1]
    contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    draw_img = frame.copy()
    # draw_img = cv2.drawContours(draw_img, contours, -1, (0, 255, 0), 2)

    for cnt in contours:
        epsilon = 0.1 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        contour_area = cv2.contourArea(cnt)
        contour_num = len(approx)
        if contour_area >= MIN_AREA and contour_num == 4:
            rotating_box = cv2.minAreaRect(approx)
            box = cv2.boxPoints(rotating_box)
            box = np.int32(box)
            box = ctrl.Reorder_Vertex(box)
            # print(box)
            warped = lb.Perspective_Transform(gray, box)
            warped = cv2.resize(warped, (200, 200), interpolation=cv2.INTER_NEAREST)
            index, location = lb.Template_Matching(warped, img, threshold=0.75)
            if index and location:
                print(f"Index: {index}, Location: {location}")

            draw_img = cv2.drawContours(draw_img, [box], -1, (0, 255, 0), 2)
    cv2.imshow('Contours', draw_img)
    cv2.imshow('Binary', binary)
    cv2.imshow('Gray', gray)
    cv2.imshow('Warped', warped)
    if cv2.waitKey(10) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()