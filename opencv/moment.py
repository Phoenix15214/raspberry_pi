import serial
import time
import cv2
import numpy as np
import pytesseract

# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def recognize_text(binary):

    # if np.mean(gray) < 127:
    #     gray = cv2.bitwise_not(gray)

    # _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 使用单字符配置
    config_single_char = '--oem 1 --psm 10 -c tessedit_char_whitelist=RLrl'# '--psm 10 -c tessedit_char_whitelist=RLrl'

    # 获取详细数据
    data = pytesseract.image_to_data(binary, lang='eng',
                                   config=config_single_char,
                                   output_type=pytesseract.Output.DICT)

    # 找到置信度最高的单个字符
    most_confident_char = ''
    highest_confidence = 0
    char_position = None

    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        confidence = int(data['conf'][i]) if data['conf'][i] else 0

        # 只考虑单个字符且置信度>0
        if len(text) == 1 and confidence > highest_confidence:
            highest_confidence = confidence
            most_confident_char = text
            char_position = (data['left'][i], data['top'][i],
                           data['width'][i], data['height'][i])

    # if most_confident_char:
    #     print(f"最明显字符: '{most_confident_char}'")
    # else:
    #     print("未找到明显的字符")

    return most_confident_char

# 生成模板 (可以根据实际字符样式调整字体、规模和厚度)
def generate_template(char, size=50, thickness=3):
    template = np.zeros((size + 20, size + 20), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5  # 调整以匹配预期字符大小
    text_size = cv2.getTextSize(char, font, font_scale, thickness)[0]
    text_x = (template.shape[1] - text_size[0]) // 2
    text_y = (template.shape[0] + text_size[1]) // 2
    cv2.putText(template, char, (text_x, text_y), font, font_scale, 255, thickness)
    contours, _ = cv2.findContours(template, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        template = template[y:y+h, x:x+w]
    return template


# 使用本地文件作为模板
image_path_L = '/home/pi/rasp_projects/l3.png'
image_path_R = '/home/pi/rasp_projects/r3.png'

# 以灰度模式读取模板图像
template_L = cv2.imread(image_path_L, cv2.IMREAD_GRAYSCALE)
template_R = cv2.imread(image_path_R, cv2.IMREAD_GRAYSCALE)

# template_L = cv2.resize(template_L, (80, 60), interpolation=cv2.INTER_LINEAR)
# template_R = cv2.resize(template_R, (80, 60), interpolation=cv2.INTER_LINEAR)


def recognize_text_opencv(gray):


    tmpl_L = template_L.copy()
    tmpl_R = template_R.copy()

    # if len(tmpl_L.shape) > 2:
    #     tmpl_L = cv2.cvtColor(tmpl_L, cv2.COLOR_BGR2GRAY)
    # if len(tmpl_R.shape) > 2:
    #     tmpl_R = cv2.cvtColor(tmpl_R, cv2.COLOR_BGR2GRAY)

    # 归一化亮度（关键）
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    tmpl_L = cv2.normalize(tmpl_L, None, 0, 255, cv2.NORM_MINMAX)
    tmpl_R = cv2.normalize(tmpl_R, None, 0, 255, cv2.NORM_MINMAX)

    method = cv2.TM_CCOEFF_NORMED

    scales = np.linspace(0.7, 1.3, 15)  # 只允许小范围缩放 → 效果更稳定

    best_L = -1
    best_R = -1

    for scale in scales:
        # Resize L
        new_L = cv2.resize(tmpl_L, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        if new_L.shape[0] > gray.shape[0] or new_L.shape[1] > gray.shape[1]:
            continue

        res = cv2.matchTemplate(gray, new_L, method)
        _, val, _, _ = cv2.minMaxLoc(res)
        best_L = max(best_L, val)

        # Resize R
        new_R = cv2.resize(tmpl_R, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        if new_R.shape[0] > gray.shape[0] or new_R.shape[1] > gray.shape[1]:
            continue

        res = cv2.matchTemplate(gray, new_R, method)
        _, val, _, _ = cv2.minMaxLoc(res)
        best_R = max(best_R, val)

    # print(f"L匹配度: {best_L:.3f}, R匹配度: {best_R:.3f}")

    # 阈值大幅提高（灰度模板匹配的正常范围是 0.5 ~ 0.9）
    threshold = 0.5

    if best_L > threshold and best_L > best_R:
        return "L"
    elif best_R > threshold and best_R > best_L:
        return "R"
    return ""


# def count_red_green_pixels_rgb(img):
#     """
#     使用RGB颜色空间计算红色和绿色像素数量（修正版）
#     """

#     if img is None:
#         print("无法加载图像，请检查")
#         return 0, 0


#     # 将BGR转换为RGB
#     rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     # 定义红色和绿色的阈值（RGB空间）
#     # 红色: R值高，G和B值低
#     # red_mask = (rgb[:, :, 0] > 140) & (rgb[:, :, 1] < 100) & (rgb[:, :, 2] < 100)

#     # # 绿色: G值高，R和B值低
#     # green_mask = (rgb[:, :, 1] > 110) & (rgb[:, :, 0] < 110) & (rgb[:, :, 2] < 110)
#     red_mask = ((rgb[:,:,0]//rgb[:,:,1]+1 > 2) & (rgb[:,:,0]//rgb[:,:,2]+1 > 2))
#     green_mask = ((rgb[:,:,1]//rgb[:,:,0]+1 > 1.5) & (rgb[:,:,1]//rgb[:,:,2]+1 > 1.5))
#     red_count = np.sum(red_mask)
#     green_count = np.sum(green_mask)
#     total_pixels = 320 * 240

#     # print(f"红色像素数量: {red_count}")
#     # print(f"绿色像素数量: {green_count}")
#     # print(f"总像素数量: {total_pixels}")
#     # print(f"红色占比: {red_count / total_pixels * 100:.2f}%")
#     # print(f"绿色占比: {green_count / total_pixels * 100:.2f}%")  # 修正了这一行

#     return red_count, green_count

def count_red_green_pixels_rgb(img):
    if img is None:
        return 0, 0
    
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 分离通道
    R = rgb[:,:,0].astype(float)
    G = rgb[:,:,1].astype(float)  
    B = rgb[:,:,2].astype(float)
    
    # 计算相对比例，避免绝对阈值
    total = R + G + B + 1e-5  # 避免除0
    
    r_ratio = R / total
    g_ratio = G / total
    b_ratio = B / total
    
    # 红色检测：R占主导，且不是白色（排除过曝）
    red_mask = (r_ratio > 0.5) & (r_ratio > g_ratio + 0.2) & (r_ratio > b_ratio + 0.2) & (total < 700)  # 排除接近白色的
    
    # 绿色检测：G占主导，且不是白色
    green_mask = (g_ratio > 0.4) & (g_ratio > r_ratio + 0.1) & (g_ratio > b_ratio + 0.1) & (total < 700)
    
    # 黄色检测：R、G占比接近，B较少
    # yellow_mask = ((r_ratio + 1e-5)/(g_ratio + 1e-5)) > 0.8 & ((r_ratio + 1e-5)/(g_ratio + 1e-5)) < 1.2 & (r_ratio > b_ratio) & (g_ratio > b_ratio)
    # yellow_mask = total > 700# (g_ratio > 0.4) & (r_ratio > 0.4) & (total < 700)
    yellow_mask = (R > 200) & (G > 160) & (B < 160)
    red_count = np.sum(red_mask)
    green_count = np.sum(green_mask)
    yellow_count = np.sum(yellow_mask)
    # test_count = np.sum(test_mask)
    
    return red_count, green_count, yellow_count


def ls(img):

    # 预处理:灰度化,二值化,取ROI
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_binary = cv2.threshold(img_gray, 60, 255, cv2.THRESH_BINARY_INV)
    height, width = img.shape[:2]
    roi_top = height // 2  # 从图像一半高度开始
    roi_bottom = height  # 到图像底部

    # 储存中心点
    center_points = []

    # 从下往上，每隔10个像素行进行一次扫描
    for y in range(roi_bottom - 1, roi_top, -10):
        # 求每一行中值为255的x坐标
        row = img_binary[y, :]
        white_pixels_x = np.where(row == 255)[0]
        # 如果找到了白色像素点
        if len(white_pixels_x) > 0:
            # 计算白色部分中心x坐标并写入center_points
            x_center = int(np.mean(white_pixels_x))
            center_points.append((x_center, y))

    # 至少要有两个点才能拟合直线
    if len(center_points) < 2:
        return 0.0  # 如果没有找到足够点，返回默认值

    # 使用np.polyfit对直线进行拟合,次数为1
    # 获得斜率slope和截距intercept
    x_coords, y_coords = zip(*center_points)
    params = np.polyfit(y_coords, x_coords, 1)
    slope = params[0]
    intercept = params[1]

    # 计算路径中心点的平均x坐标与中间位置的偏移
    avg_x = int(np.mean(x_coords)) - width // 2

    # 计算直线的倾斜角度
    angle_rad = np.arctan(slope)  # 得到弧度
    angle_deg = np.degrees(angle_rad)  # 转换为度

    # 绘制找到的中心点（绿色）与拟合得到的直线
    vis_image = cv2.cvtColor(img_binary, cv2.COLOR_GRAY2BGR)
    for point in center_points:
        cv2.circle(vis_image, point, 5, (0, 255, 0), -1)
    y1 = roi_bottom
    x1 = int(slope * y1 + intercept)
    y2 = roi_top
    x2 = int(slope * y2 + intercept)
    cv2.line(vis_image, (x1, y1), (x2, y2), (255, 0, 0), 3)

    # 在图像上显示计算出的角度和平均x坐标
    cv2.putText(vis_image, f'Angle: {angle_deg:.2f} degrees', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(vis_image, f'Avg X: {avg_x}', (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    return angle_deg

def count_white_pixels_at_y(binary_img, y):
        try:
                # 获取指定行的所有像素
            row = binary_img[y, :]
                # 统计白色像素（值为255）
            white_count = np.sum(row == 255)
            return white_count
        except Exception as e:
            print(f"错误: {e}")
            return 0

def get_stop(binary_img, roi_height):
    state = 0
    # last_state = 0
    count = 0
    line_height = 0
    if count_white_pixels_at_y(binary_img, roi_height * 2 // 3) >= 80:
        for i in range(roi_height):
            white_pixels = count_white_pixels_at_y(binary_img, i)
            state = 1 if white_pixels >= 80 else 0
            if state == 1:
                line_height += 1
            else:
                if line_height >= 5:
                    count += 1
                    line_height = 0
                else:
                    line_height = 0
            #     line_height = 0
            # if (line_height >= 5 and state == 0):
            #     count += 1
            # count += 1 if last_state == 0 and state ==1 else 0
            # last_state = state
    if line_height >= 5:
        count += 1
    if count == 0:
        return 0
    elif count == 1:
        return 1
    else:
        return 2


def detect_horizontal_line_in_region(binary_img, min_white=160, max_white = 5, min_length=5):
    """
    检测一个区域中是否存在有效横线（连续 >= min_length 行，每行白像素 >= min_white）
    """
    if binary_img.size == 0:
        return False

    white_pixels = (binary_img > 0).sum(axis=1)
    is_white_line = white_pixels >= min_white
    is_black = white_pixels < max_white
    # 找连续段
    padded = np.concatenate(([0], is_white_line.astype(int), [0]))
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    lengths = ends - starts
    
    return np.any(lengths >= min_length)

def get_stop_dynamic(binary_img, total_roi_height=100, split_ratio=0.5):
    """
    动态场景下识别单根/双根横线
    :param binary_img: 二值图（H x W），OpenCV 输出
    :param total_roi_height: 总ROI高度（建议取图像底部区域）
    :param split_ratio: 分割比例，0.5 表示上下平分
    :return: 0=无, 1=单根（下区）, 2=双根（上下都有）
    """
    h, w = binary_img.shape
    start_y = max(0, h - total_roi_height)
    roi = binary_img[start_y:start_y + total_roi_height]

    split_idx = int(total_roi_height * split_ratio)
    lower_roi = roi[:split_idx]      # 靠近图像底部（先看到）
    upper_roi = roi[split_idx:]      # 靠近图像顶部（后看到）

    has_upper = detect_horizontal_line_in_region(upper_roi)
    if(has_upper):
        has_lower = detect_horizontal_line_in_region(lower_roi)
    else:
        has_lower = False
    # has_upper = detect_horizontal_line_in_region(upper_roi)

    if has_lower and has_upper:
        return 2  # 两根都看到
    elif has_lower or has_upper:
        return 1  # 只看到一根（优先认为是进入状态）
    else:
        return 0


def get_center_point(img):
    img_output = img.copy()
    min_area_threshold = 100  # 降低最小面积阈值（根据实际调整）

    # 1. 灰度转换与二值化（优化）
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 使用自适应阈值（适应光照变化）
    # threshold_value = 80
    # img_binary = cv2.adaptiveThreshold(
    #     img_gray, 255,cv2.ADAPTIVE_THRESH_MEAN_C,
    #     cv2.THRESH_BINARY_INV, 11, 2
    # )

    # 使用固定阈值（需要手动调整阈值）
    threshold_value = 80  # 阈值，范围0-255
    _, img_binary = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_BINARY)

    # 2. 提取所有轮廓（包括内部）
    cnts = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    cx, cy = -1, -1
    is_junction = 0  # 0: 无岔路，1: 有岔路

    if len(cnts) > 0:
        #print(f"原始轮廓数量: {len(cnts)}")  # 调试信息

        # 3. 计算中心点（主路径）
        largest_cnt = max(cnts, key=cv2.contourArea)
        m = cv2.moments(largest_cnt)
        if m['m00'] > 0:
            cx = int(m['m10'] / m['m00'])
            cy = int(m['m01'] / m['m00'])
            cv2.circle(img_output, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(img_output, f'Avg X: {cx}', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # 4. 筛选有效轮廓（面积>最小阈值）
        cnts_sorted = sorted(cnts, key=cv2.contourArea, reverse=True)
        main_contours = [cnt for cnt in cnts_sorted if cv2.contourArea(cnt) > min_area_threshold]
        #print(f"有效轮廓数量: {len(main_contours)}")  # 调试信息

        # 5. 判断是否为岔路（有效轮廓≥2）
        is_junction = 1 if len(main_contours) > 2 else 0
        #print(f"is_junction: {is_junction}")  # 调试信息

        # 6. 绘制所有轮廓（可视化）
        cv2.putText(img_output, f'contour: {len(cnts)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        for cnt in cnts:
            cv2.drawContours(img_output, cnt, -1, (0, 255, 0), 3)



    return cx, cy, img_binary, is_junction


class SerialPacket:
    def __init__(self, port="COM6", baudrate=115200, timeout=0.1):
        """初始化串口"""
        try:
            self.ser = serial.Serial(port, baudrate, timeout=timeout)
            if self.ser.is_open:
                print(f"串口 {port} 已成功打开！")
            else:
                print(f"串口 {port} 打开失败！")
        except serial.SerialException as e:
            print(f"串口 {port} 打开失败，错误信息：{e}")
        except Exception as e:
            print(f"发生未知错误：{e}")
        # 初始化包头和包尾
        self.header = bytearray([0xFF, 0xAA])
        self.tail = bytearray([0x55, 0xFE])
        self.data = bytearray()
        self.index = 0  # 数据插入位置

    def __clear_packet(self):
        """清空包数据，只保留包头"""
        self.data = bytearray()
        self.index = 0  # 数据插入位置清零

    def insert_byte(self, value):
        """在包头后插入单个字节"""
        self.data.insert(self.index, value)  # 插入到数据部分开头（包头之后）
        self.index += 1

    def insert_two_bytes(self, values):
        """在包头后插入两个字节"""
        self.data.insert(self.index, values[0])  # 插入到数据部分开头（包头之后）
        self.index += 1
        self.data.insert(self.index, values[1])  # 插入到数据部分开头（包头之后）
        self.index += 1

    def insert_bytes(self, index, values):
        """在指定位置插入多个字节"""
        for i, val in enumerate(values):
            self.data.insert(index + i, val)
            self.index += 1

    def num_to_bytes(self, value):
        """发送16位整数并拆分为高8位和低8位"""
        if not 0 <= value <= 0xFFFF:
            raise ValueError("输入值必须在 0~65535 之间")

        high_byte = (value >> 8) & 0xFF  # 高8位
        low_byte = value & 0xFF  # 低8位

        return [high_byte, low_byte]

    def __build_packet(self):
        """生成完整数据包"""
        return self.header + self.data + self.tail

    def send_packet(self):
        """发送完整数据包"""
        # 构建帧头 + 数据包 + 帧尾
        packet = self.__build_packet()
        # 发送数据包
        self.ser.write(packet)

        # # 十六进制美化输出
        # hex_str = ' '.join([f'{b:02X}' for b in packet])
        # print(f"发送: {hex_str}")

        # 清空数据包
        self.__clear_packet()


