import cv2
import numpy as np

# 傅里叶变换有关参数
FFT_HIGHPASS = 0
FFT_LOWPASS = 1

# 颜色提取相关参数
RED, GREEN, BLUE, WHITE, YELLOW, ORANGE, PURPLE, PINK, BROWN, GRAY = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

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

BLUE_LOWER = np.array([100, 43, 46])
BLUE_UPPER = np.array([124, 255, 255])

ORANGE_LOWER = np.array([11, 43, 46])
ORANGE_UPPER = np.array([25, 255, 255])

PURPLE_LOWER = np.array([125, 43, 46])
PURPLE_UPPER = np.array([155, 255, 255])

PINK_LOWER = np.array([160, 50, 100])
PINK_UPPER = np.array([180, 150, 220])

BROWN_LOWER = np.array([10, 100, 50])
BROWN_UPPER = np.array([20, 200, 150])

GRAY_LOWER = np.array([0, 0, 50])
GRAY_UPPER = np.array([180, 50, 200])



# 傅里叶变换有关函数
def FFT_Filtering(img, radius=30, mode=FFT_HIGHPASS):
    """
    对输入图像进行高通滤波处理，增强图像的边缘和细节。默认高通滤波。
    参数:
        img: 输入图像，灰度图与彩色图均可。
        radius: 高通滤波器的半径，默认为30。
    返回:
        处理后的图像。
    """
    if img is None:
        raise ValueError("输入图像不能为空")
    if img.ndim not in [2, 3]:
        raise ValueError("输入必须为彩色图或灰度图")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    
    rows, cols = img.shape
    crow, ccol = rows // 2 , cols // 2
    
    if(radius < 0):
        raise ValueError("半径必须为正整数")
    elif radius == 0:
        radius = 1
    elif(radius > min(crow, ccol)):
        raise ValueError("半径过大，无法应用滤波器")
    
    img_float32 = np.float32(img)
    
    dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    if mode == FFT_LOWPASS:
        mask = np.zeros((rows, cols, 2), np.uint8)
        cv2.circle(mask, (ccol, crow), radius, (1, 1, 1), -1)
    else:
        mask = np.ones((rows, cols, 2), np.uint8)
        cv2.circle(mask, (ccol, crow), radius, (0, 0, 0), -1)

    fshift = dft_shift*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    img_back = np.uint8(img_back)

    return img_back

# 模板匹配有关函数
def Template_Matching(img, templates, threshold=0.5, min_scale=0.7,num_scale=5, method=cv2.TM_CCOEFF_NORMED):
    """
    在输入图像中查找与模板图像匹配的区域。
    参数:
        img: 输入图像，灰度图与彩色图均可。
        template: 模板图像，必须为灰度图。
        threshold:模板匹配阈值,默认为0.5。
        min_scale:在多尺度匹配中最小尺度,默认为0.7。
        num_scale:多尺度匹配中的尺度数量,默认为5。
        method: 模板匹配方法,默认为cv2.TM_CCOEFF_NORMED。
    返回:
        最佳匹配结果在模板中的索引。
    """
    
    best_template = []
    best_index = 0
    if img is None:
        raise ValueError("输入图像不能为空")
    if img.ndim not in [2, 3]:
        raise ValueError("输入图像必须为灰度图或彩色图")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    for template in templates:
        template = cv2.normalize(template, None, 0, 255, cv2.NORM_MINMAX)

    scales = np.linspace(min_scale, 1.0, num_scale)

    for template in templates:
        best_match = -1
        for scale in scales:
            template = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
            _, val, _, _ = cv2.minMaxLoc(res)
            best_match = max(val, best_match)
        best_template.append(best_match)

    best_index = np.argmax(best_template)

    return best_index if best_template[best_index] > threshold else -1

def Create_Arr(*args):
    """
    创建任意类型的数组。
    参数:
        *args:任意数量的同类型元素。
    返回:
        合成后的数组。
    """
    if args is None:
        raise ValueError("输入不能为空")
    elements = []
    for element in args:
        elements.append(element)

    return elements

# 颜色提取相关函数
def Color_Extraction(img, color=RED):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if color == RED:
        mask = cv2.inRange(img_hsv, RED_LOWER1, RED_UPPER1) + cv2.inRange(img_hsv, RED_LOWER2, RED_UPPER2)
    elif color == GREEN:
        mask = cv2.inRange(img_hsv, GREEN_LOWER, GREEN_UPPER)
    elif color == YELLOW:
        mask = cv2.inRange(img_hsv, YELLOW_LOWER, YELLOW_UPPER)
    elif color == WHITE:
        mask = cv2.inRange(img_hsv, WHITE_LOWER, WHITE_UPPER)
    elif color == BLUE:
        mask = cv2.inRange(img_hsv, BLUE_LOWER, BLUE_UPPER)
    elif color == ORANGE:
        mask = cv2.inRange(img_hsv, ORANGE_LOWER, ORANGE_UPPER)
    elif color == PURPLE:
        mask = cv2.inRange(img_hsv, PURPLE_LOWER, PURPLE_UPPER)
    elif color == PINK:
        mask = cv2.inRange(img_hsv, PINK_LOWER, PINK_UPPER)
    elif color == BROWN:
        mask = cv2.inRange(img_hsv, BROWN_LOWER, BROWN_UPPER)
    elif color == GRAY:
        mask = cv2.inRange(img_hsv, GRAY_LOWER, GRAY_UPPER)
    else:
        raise ValueError("不支持的颜色类型")

    result = cv2.bitwise_and(img, img, mask=mask)

    return result

def color_extraction_dynamic(img, hsv_lower, hsv_upper):
    img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img_hsv, hsv_lower, hsv_upper)
    result = cv2.bitwise_and(img, img, mask=mask)

    return result


# PID相关函数

class PID_Inc():
    def __init__(self, kp, ki, kd, goal, frquency):
        self.frequency = frquency
        self.kp = kp
        self.ki = ki / self.frequency
        self.kd = kd * self.frequency
        self.goal = goal
        self.output_integral = 0.0
        self.integral_limit = 2000
        self.prev_error = 0.0
        self.prev_prev_error = 0.0

    def Set_Goal(self, goal):
        self.goal = goal

    def Set_Integral_Limit(self, limit):
        self.integral_limit = limit

    def Cal_PID(self, current_val):
        error = self.goal - current_val
        output = self.kp * (error - self.prev_error) + self.ki * error + self.kd * (error - 2 * self.prev_error + self.prev_prev_error)
        self.prev_prev_error = self.prev_error
        self.prev_error = error
        self.output_integral += output

        return self.output_integral if self.integral_limit is None else max(-self.integral_limit, min(self.integral_limit, self.output_integral))

class PID_Loc():
    def __init__(self, kp, ki, kd, goal, frequency):
        self.frequency = frequency
        self.kp = kp
        self.ki = ki / self.frequency
        self.kd = kd * self.frequency
        self.goal = goal
        self.integral_limit = 1000
        self.output_limit = 2000
        self.integral = 0.0
        self.prev_error = 0.0

    def Set_Goal(self, goal):
        self.goal = goal
    
    def Set_Limit(self, limit):
        self.integral_limit = limit

    def Set_Output_Limit(self, limit):
        self.output_limit = limit

    def Cal_PID(self, current_val):
        error = self.goal - current_val
        derivative = error - self.prev_error
        self.prev_error = error
        temp_output = self.kp * error + self.kd * derivative
        should_integrate = True
        if temp_output + self.ki * self.integral > self.output_limit and error > 0:
            should_integrate = False
        elif temp_output + self.ki * self.integral < -self.output_limit and error < 0:
            should_integrate = False
        if should_integrate:
            self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral + error))
        output = temp_output + self.ki * self.integral
        
        return max(-self.output_limit, min(self.output_limit, output))
