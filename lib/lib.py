import cv2
import numpy as np

# 傅里叶变换有关函数
def high_freq_filtering(img, radius=30):
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
        raise ValueError("半径过大，无法应用高通滤波器")
    
    img_float32 = np.float32(img)
    
    dft = cv2.dft(np.float32(img_float32))
    dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    mask = np.ones((rows, cols, 2), np.uint8)
    cv2.circle(mask, (ccol, crow), radius, (0, 0, 0), -1)

    fshift = dft_shift*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    img_back = np.uint8(img_back)

    return img_back

def low_freq_filtering(img, radius=30):
    """
    对输入图像进行低通滤波处理，去除图像的高频噪声。
    参数:
        img: 输入图像，灰度图与彩色图均可。
        radius: 低通滤波器的半径，默认为30。
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
        raise ValueError("半径过大，无法应用低通滤波器")
    
    img_float32 = np.float32(img)
    
    dft = cv2.dft(np.float32(img_float32))
    dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    mask = np.zeros((rows, cols, 2), np.uint8)
    cv2.circle(mask, (ccol, crow), radius, (1, 1, 1), -1)

    fshift = dft_shift*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    img_back = np.uint8(img_back)

    return img_back


    
    