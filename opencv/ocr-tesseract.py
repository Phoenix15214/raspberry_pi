import cv2
import numpy as np
from moment import SerialPacket, count_red_green_pixels_rgb
import cv2
import numpy as np
from moment import recognize_text

def process_image(img, threshold_value=51):
    """
    处理输入图像，返回二值化图像。
    """
    # 灰度转换与二值化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)# cv2.THRESH_BINARY_INV)
    return binary

# 初始化摄像头
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 创建窗口和滑块
cv2.namedWindow("Binary_0")
cv2.createTrackbar("Threshold", "Binary_0", 31, 255, lambda x: None)

pack = SerialPacket(port="/dev/ttyUSB0", baudrate=115200, timeout=0.1)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        height, width = frame.shape[:2]
        roi = frame[height//2 - 60: height//2 + 60, width//2 - 80: width//2 + 80].copy()

        # 获取阈值
        threshold_value = cv2.getTrackbarPos("Threshold", "Binary_0")

        # 计算红绿灯状态和字符识别结果
        red_count, green_count = count_red_green_pixels_rgb(roi)
        rgb_control = 1 if red_count > 3000 else (2 if green_count > 3000 else 0)
        binary = process_image(roi, threshold_value)
        text = recognize_text(binary)
        str_control = 1 if text == 'L' else (2 if text == 'R' else 0)

        print(f"红绿灯判断：{rgb_control}, 字符识别:{str_control}")

        # 显示处理后的灰度图
        cv2.imshow("Binary_0", binary)

        # 发送数据包
        pack.insert_byte(0x04)
        pack.insert_two_bytes(pack.num_to_bytes(rgb_control))
        pack.insert_two_bytes(pack.num_to_bytes(str_control))
        # pack.send_packet()
        if cv2.waitKey(1) & 0xFF == 27:  # ESC退出
            break

except KeyboardInterrupt:
    print("程序中断")

finally:
    cap.release()
    cv2.destroyAllWindows()
    