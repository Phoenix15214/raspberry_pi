import cv2
from moment import get_center_point

img = cv2.imread("test1.jpg")
height, width = img.shape[:2]
roi = img[height // 2:, :]
roi = cv2.imread('roi.jpg')
# roi = cv2.bitwise_and(img, img, mask=roi_mask)
x, y, roi = get_center_point(roi)
cv2.imshow("roi", roi)
cv2.waitKey(0)
