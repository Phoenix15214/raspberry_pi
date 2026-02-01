import cv2
import numpy as np
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# HSV范围
RED_LOWER1 = np.array([0, 43, 46])
RED_UPPER1 = np.array([10, 255, 255])
RED_LOWER2 = np.array([156, 43, 46])
RED_UPPER2 = np.array([180, 255, 255])

WHITE_LOWER = np.array([0, 0, 221])
WHITE_UPPER = np.array([180, 30, 255])

GREEN_LOWER = np.array([35, 43, 46])
GREEN_UPPER = np.array([77, 255, 255])

YELLOW_LOWER = np.array([26, 43, 46])
YELLOW_UPPER = np.array([34, 255, 255])

if __name__ == '__main__':
    # 读取图像
    img = cv2.imread("test2.jpg")
    # 转换为HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 提取
    red_mask = cv2.inRange(img_hsv, RED_LOWER1, RED_UPPER1) + cv2.inRange(img_hsv, RED_LOWER2, RED_UPPER2)
    white_mask = cv2.inRange(img_hsv, WHITE_LOWER, WHITE_UPPER)
    green_mask = cv2.inRange(img_hsv, GREEN_LOWER, GREEN_UPPER)
    yellow_mask = cv2.inRange(img_hsv, YELLOW_LOWER, YELLOW_UPPER)
    #
    red = cv2.bitwise_and(img, img, mask=red_mask)
    green = cv2.bitwise_and(img, img, mask=green_mask)
    white = cv2.bitwise_and(img, img, mask=white_mask)
    yellow = cv2.bitwise_and(img, img, mask=yellow_mask)

    stack = np.hstack((img, cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR), red))
    cv2.imshow("red", stack)
    cv2.waitKey(0)




