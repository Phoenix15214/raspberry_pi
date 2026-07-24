import cv2
import numpy as np

def main():
    # 打开摄像头（默认设备 0）
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    print("按 'q' 键退出程序")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取视频帧")
            break

        # 转为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 二值化：使用固定阈值（可调整）
        # 这里采用 Otsu 自动阈值，也可改用固定值如 128
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.dilate(binary, kernel, iterations=2) 
        # 或者使用自适应阈值：
        # binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                                cv2.THRESH_BINARY, 11, 2)

        # 将二值图像转为 float32，供 cornerHarris 使用
        binary_float = np.float32(binary)

        # Harris 角点检测
        # blockSize: 邻域大小, ksize: Sobel 算子孔径, k: Harris 自由参数
        dst = cv2.cornerHarris(binary_float, blockSize=2, ksize=3, k=0.04)

        # 膨胀以便清晰标记角点
        dst = cv2.dilate(dst, None)

        # 在原图上标记角点（红色圆点）
        # 阈值设定为响应最大值的 1%
        frame[dst > 0.01 * dst.max()] = [0, 0, 255]

        # 显示结果
        cv2.imshow('Original with Corners', frame)
        cv2.imshow('Binary Image', binary)

        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()