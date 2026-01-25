import cv2
import numpy as np
from moment import SerialPacket, count_red_green_pixels_rgb
from moment import recognize_text_opencv
from moment import get_stop_dynamic

def get_center_point(img, min_area_threshold = 40, threshold_value=51):
    """
    输入：
        img: BGR图像
        threshold_value: 二值化阈值
    输出：
        cx, cy: 道路中心点坐标
        img_binary: 二值化图像
        img_output: 可视化图像（带中心点和中心线）
    """
    img_output = img.copy()

    # 1. 灰度化
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 反二值化，黑色道路变白
    _, img_binary = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
    # _, img_binary_0 = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_BINARY)

    # 3. 查找轮廓
    cnts = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    cx, cy = -1, -1

    is_junction = 0

    if len(cnts) > 0:
        # 找最大轮廓
        largest_cnt = max(cnts, key=cv2.contourArea)
        m = cv2.moments(largest_cnt)
        if m['m00'] > 0:
            cx = int(m['m10'] / m['m00'])
            cy = int(m['m01'] / m['m00'])
            # 绘制中心点
            cv2.circle(img_output, (cx, cy), 5, (0, 255, 0), -1)
            # 绘制纵向中心线
            cv2.line(img_output, (cx, 0), (cx, img_output.shape[0]), (255, 0, 0), 2)
        # 筛选有效轮廓
        cnts_sorted = sorted(cnts, key=cv2.contourArea, reverse=True)
        main_contours = [cnt for cnt in cnts_sorted if cv2.contourArea(cnt) > min_area_threshold]
        is_junction = 1 if len(main_contours) > 2 else 0

    return cx, cy, img_binary,img_gray, img_output# , is_junction


def process_image(img):# , threshold_value=51):
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

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 创建窗口和滑块
cv2.namedWindow("Binary_0")
cv2.namedWindow("ROI")
# cv2.createTrackbar("Threshold1", "Binary_0", 100, 255, lambda x: None)
cv2.createTrackbar("Threshold", "Binary_0", 100, 255, lambda x: None)

pack = SerialPacket(port="/dev/ttyUSB0", baudrate=115200, timeout=0.1)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        height, width = frame.shape[:2]
        roi = frame[height//2 - 120: height//2 + 120, width//2 - 160: width//2 + 160].copy()
        roi1 = frame[height//2 - 60: height//2 + 60, width//2 - 80: width//2 + 80].copy()
        # 获取阈值
        threshold_value1 = cv2.getTrackbarPos("Threshold", "Binary_0")
        # threshold_value2 = cv2.getTrackbarPos("Threshold2", "Binary_0")
        # 计算红绿灯状态和字符识别结果
        red_count, green_count, yellow_count = count_red_green_pixels_rgb(roi)
        rgb_control = 1 if red_count > 2000 else (2 if green_count > 1000 else(3 if yellow_count > 3000 else 0) )
        # binary, gray = process_image(roi, threshold_value1)
        cx, cy, binary, img_gray, output = get_center_point(roi, 40, threshold_value1)
        gray = process_image(roi1)
        text = recognize_text_opencv(gray)
        str_control = 1 if text == 'L' else (2 if text == 'R' else 0)
        roi_height, roi_width = roi.shape[:2]
        is_stop = get_stop_dynamic(binary, roi_height)
        print(is_stop)
        print(cx)
        # print(f"红绿灯判断：{rgb_control}, 字符识别:{str_control}")

        # 显示处理后的灰度图
        cv2.imshow("Binary_0", binary)
        # cv2.imshow("ROI", roi1)
        # 发送数据包
        pack.insert_byte(0x08)
        pack.insert_two_bytes(pack.num_to_bytes(rgb_control))
        pack.insert_two_bytes(pack.num_to_bytes(str_control))
        pack.insert_two_bytes(pack.num_to_bytes(is_stop))
        pack.insert_two_bytes(pack.num_to_bytes(cx + 100))
        pack.send_packet()
        if cv2.waitKey(1) & 0xFF == 27:  # ESC退出
            break

except KeyboardInterrupt:
    print("程序中断")

finally:
    cap.release()
    cv2.destroyAllWindows()
    