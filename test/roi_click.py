import cv2
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


# 全局变量，用于存储点击的坐标和点击计数
drawing = False  # 标志，表示是否正在选择区域
points = []  # 存储点击的 (x, y) 坐标
WINDOW_NAME = 'Select ROI'


def generate_roi_mask(img):

    global drawing, points, WINDOW_NAME

    # 获取图像尺寸
    height, width = img.shape[:2]

    # 创建一个可修改的副本用于显示点击标记
    display_img = img.copy()

    # 初始化全局变量
    drawing = False
    points = []

    # 2. 鼠标回调函数
    def mouse_callback(event, x, y, flags, param):
        nonlocal display_img, img

        # 鼠标左键按下事件
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 2:
                points.append((x, y))

                # 在点击位置绘制一个小圆圈作为标记
                cv2.circle(display_img, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow(WINDOW_NAME, display_img)

                if len(points) == 1:
                    print(f"第一次点击坐标: ({x}, {y}). 请点击第二个点...")
                elif len(points) == 2:
                    # 3. 两个点已选定，退出循环
                    print(f"第二次点击坐标: ({x}, {y}). ROI选择完成。")
                    # 设置一个标志让主循环退出
                    global drawing
                    drawing = True

                    # 绘制最终矩形 (仅用于展示)
                    cv2.rectangle(display_img, points[0], points[1], (0, 255, 0), 2)
                    cv2.imshow(WINDOW_NAME, display_img)

    # 3. 设置窗口和鼠标回调
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    print("\n--- 操作提示 ---")
    print("请在窗口中依次点击鼠标左键两次，以确定矩形的两个对角点。")

    # 循环等待用户点击完成
    while not drawing:
        cv2.imshow(WINDOW_NAME, display_img)
        # 等待按键，1毫秒
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("用户中断操作。")
            break

    cv2.destroyAllWindows()

    # 4. 生成 Mask 并返回
    if len(points) == 2:
        # 创建一个全零的掩码 (与原图同样的高度和宽度)
        mask = np.zeros((height, width), dtype=np.uint8)

        # 确保坐标顺序正确 (左上角, 右下角)
        p1 = points[0]
        p2 = points[1]

        x_min = min(p1[0], p2[0])
        x_max = max(p1[0], p2[0])
        y_min = min(p1[1], p2[1])
        y_max = max(p1[1], p2[1])

        roi = img[y_min:y_max, x_min:x_max]
        cv2.imwrite(os.getcwd() + '/roi.jpg', roi)

        # 将矩形区域设置为 255
        mask[y_min:y_max, x_min:x_max] = 255

        print("\nROI Mask 已成功生成。")
        return mask
    else:
        print("\n未完成 ROI 选择，返回 None。")
        return None


# --- 示例运行 ---
if __name__ == '__main__':

    img = cv2.imread("test1.jpg")
    # 调用函数
    roi_mask = generate_roi_mask(img)

    if roi_mask is not None:
        # 显示生成的 Mask
        cv2.imshow("Generated ROI Mask", roi_mask)

        # 可选：将 Mask 应用于原图，展示提取效果
        # 将 Mask 从单通道扩展为三通道以便与彩色图进行按位与操作
        mask_bgr = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)
        extracted_roi = cv2.bitwise_and(img, img, mask=roi_mask)
        cv2.imshow("Extracted ROI", extracted_roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite(os.getcwd() + '/roi_mask.jpg', roi_mask)