import cv2
import numpy as np
from moment import SerialPacket, count_red_green_pixels_rgb
from moment import recognize_text_opencv
import time

def process_image(img, threshold_value=51):
    """
    处理输入图像，返回二值化图像。
    """
    # 灰度转换与二值化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
    return gray

# 初始化摄像头
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
count = 0
reset = False
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 创建窗口和滑块
# cv2.namedWindow("Binary_0")
# cv2.namedWindow("ROI2")
# cv2.createTrackbar("Threshold", "Binary_0", 120, 255, lambda x: None)

pack = SerialPacket(port="/dev/ttyUSB0", baudrate=115200, timeout=0.1)
fps = 0
current_time = time.time()
last_time = current_time
smooth_fps = 0
alpha = 0.1

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        height, width = frame.shape[:2]
        roi1 = frame[height//2 - 60: height//2 + 60, width//2 - 80: width//2 + 80].copy()
        roi2 = frame[height//2 - 60: height//2 + 60, width//2 - 160 : width//2 + 160].copy()
        roi1 = cv2.resize(roi1, (80, 60), interpolation=cv2.INTER_LINEAR)
        # 获取阈值
        # threshold_value = cv2.getTrackbarPos("Threshold", "Binary_0")
        current_time = time.time()
        fps = 1 / (current_time - last_time)
        last_time = current_time
        smooth_fps = (1 - alpha) * smooth_fps + alpha * fps
        # 计算红绿灯状态和字符识别结果
        red_count, green_count, yellow_count = count_red_green_pixels_rgb(roi2)
        # print(f"Red: {red_count}, Green: {green_count}, Yellow: {yellow_count}, FPS: {smooth_fps:.2f}")
        rgb_control = 1 if red_count > 1000 else (2 if green_count > 1000 else(3 if yellow_count > 1000 else 0) )
        gray = process_image(roi1,120)#  threshold_value)
        text = recognize_text_opencv(gray)
        # 识别到R时返回1，识别到L时返回2，其他时候返回0
        str_control = 1 if text == 'R' else (2 if text == 'L' else 0)
        # print(f"Red: {red_count}, Green: {green_count}, Yellow: {yellow_count}, FPS: {smooth_fps:.2f},红绿灯判断：{rgb_control}, 字符识别:{str_control}")
        # print(f"红绿灯判断：{rgb_control}, 字符识别:{str_control}")
        # print(f" FPS: {smooth_fps:.2f},红绿灯判断：{rgb_control}, 字符识别:{str_control}")
        if rgb_control != 0 or str_control != 0:
            # print(f"红绿灯判断：{rgb_control}, 字符识别:{str_control}")
            print(f" FPS: {smooth_fps:.2f},红绿灯判断：{rgb_control}, 字符识别:{str_control}")
            count += 1
            print(count)
            if str_control == 0 or rgb_control == 0:
                reset = True      
        else:
            count = 0
            reset = False

        # 显示处理后的灰度图
        cv2.putText(gray, f"FPS: {smooth_fps:.2f}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
        cv2.imshow("Binary_0", gray)
        # cv2.imshow("ROI2", roi2)
        # 发送数据包
        pack.insert_byte(0x08)
        pack.insert_two_bytes(pack.num_to_bytes(rgb_control))
        pack.insert_two_bytes(pack.num_to_bytes(str_control))
        pack.insert_two_bytes(pack.num_to_bytes(0))
        pack.insert_two_bytes(pack.num_to_bytes(0))
        # pack.send_packet()
        if cv2.waitKey(1) & 0xFF == 27:  # ESC退出
            break

except KeyboardInterrupt:
    print("程序中断")

finally:
    cap.release()
    cv2.destroyAllWindows()
    