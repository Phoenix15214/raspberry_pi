import cv2
import numpy as np

# 傅里叶变换有关参数
FFT_HIGHPASS, FFT_LOWPASS = 0, 1


# 傅里叶变换有关函数
def fft_filtering(img, radius=30, mode=FFT_HIGHPASS):
    """
    对输入图像进行高通滤波处理，增强图像的边缘和细节。
    参数:
        img: 输入图像，灰度图与彩色图均可。
        radius: 高通滤波器的半径，默认为30。
    返回:
        处理后的图像。
    """
    if(len(img.shape) == 3):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif(len(img.shape) == 2):
        pass
    else:
        raise ValueError("输入图像必须为灰度图或彩色图")
    
    rows, cols = img.shape
    crow, ccol = rows // 2 , cols // 2
    
    if(radius <= 0):
        raise ValueError("半径必须为正整数")
    elif(radius > min(crow, ccol)):
        raise ValueError("半径过大，无法应用滤波器")
    
    img_float32 = np.float32(img)
    
    dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    if mode not in [FFT_HIGHPASS, FFT_LOWPASS]:
        raise ValueError("mode参数必须为FFT_HIGHPASS或FFT_LOWPASS")
    if mode == FFT_HIGHPASS:
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
def template_matching(img, templates, min_scale=0.7,num_scale=5, method=cv2.TM_CCOEFF_NORMED):
    """
    在输入图像中查找与模板图像匹配的区域。
    参数:
        img: 输入图像，灰度图与彩色图均可。
        template: 模板图像，必须为灰度图。
        method: 模板匹配方法，默认为cv2.TM_CCOEFF_NORMED。
    返回:
        匹配结果图像和匹配位置的左上角坐标。
    """
    best_template = []
    best_index = 0

    
    if (img.ndim not in [2, 3]):
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

    return best_index
