import cv2
import numpy as np


# 回调函数不做任何操作
def callback(x):
    pass


# 读取图片
img = cv2.imread('test1.jpg')

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 创建窗口
window_name = 'Binary Threshold Adjuster'
cv2.namedWindow(window_name)

# 创建滑条
cv2.createTrackbar('Threshold', window_name, 127, 255, callback)

if __name__ == '__main__':

    while True:
        # 从滑条获取当前的阈值
        current_threshold = cv2.getTrackbarPos('Threshold', window_name)

        # 二值化
        ret, img_binary = cv2.threshold(img_gray, current_threshold, 255, cv2.THRESH_BINARY_INV)

        # 显示图像
        stack = np.hstack((img, cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR),
                           cv2.cvtColor(img_binary, cv2.COLOR_GRAY2BGR)))
        cv2.imshow(window_name, stack)

        # q退出
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

