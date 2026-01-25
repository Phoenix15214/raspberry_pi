import cv2
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# 回调函数不做任何操作
def callback(x):
    pass


# 读取图片
img = cv2.imread('test_canny.png')

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 创建窗口
window_name = 'canny'
cv2.namedWindow(window_name)

# 创建滑条
cv2.createTrackbar('Threshold1', window_name, 60, 255, callback)
cv2.createTrackbar('Threshold2', window_name, 150, 255, callback)

if __name__ == '__main__':

    while True:

        current_threshold1 = cv2.getTrackbarPos('Threshold1', window_name)
        current_threshold2 = cv2.getTrackbarPos('Threshold2', window_name)
        edges = cv2.Canny(img_gray, current_threshold1, current_threshold2)
        # 显示图像
        stack = np.hstack((img,cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)))
        cv2.imshow("www", img_gray)
        cv2.imshow(window_name, stack)

        # q退出
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
