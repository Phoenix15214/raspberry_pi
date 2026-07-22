import cv2
import numpy as np

# 全局变量，用于存储鼠标点击的坐标
click_x, click_y = -1, -1
clicked = False
CAMERA_FPS = 30
CAMERA_WIDTH = 1280 # 1080p 1920*1080
CAMERA_HEIGHT = 720 # 1080p 1920*1080

def mouse_callback(event, x, y, flags, param):
    """鼠标回调函数：记录左键点击的坐标"""
    global click_x, click_y, clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        click_x, click_y = x, y
        clicked = True

def open_camera():
    try:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        return cap
    except Exception as e:
        print(f"Error opening camera: {e}")
        raise RuntimeError("Failed to open camera.")
        return None

def main():
    global click_x, click_y, clicked

    # 打开默认摄像头（索引 0）
    cap = open_camera()
    if not cap.isOpened():
        print("无法打开摄像头，请检查设备连接")
        return

    # 创建窗口并绑定鼠标回调
    window_name = "Camera - Click to get HSV"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("按 'q' 键退出程序")
    print("在画面中点击鼠标左键，控制台将输出该位置的 HSV 值")

    while True:
        ret, frame = cap.read()
        # frame = frame[CAMERA_HEIGHT//4:3*CAMERA_HEIGHT//4, CAMERA_WIDTH//4:3*CAMERA_WIDTH//4]
        # frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT), interpolation=cv2.INTER_NEAREST)
        if not ret:
            print("无法读取帧")
            break

        # 如果鼠标点击过，则处理点击坐标
        if clicked:
            # 确保坐标在图像范围内
            h, w = frame.shape[:2]
            if 0 <= click_x < w and 0 <= click_y < h:
                # 获取该像素的 BGR 值
                bgr = frame[click_y, click_x]
                # 转换为 HSV（注意 OpenCV 使用 H:0-179, S:0-255, V:0-255）
                hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
                print(f"点击位置 ({click_x}, {click_y}) 的 HSV 值: H={hsv[0]}, S={hsv[1]}, V={hsv[2]}")
            else:
                print(f"点击位置 ({click_x}, {click_y}) 超出图像范围")
            # 重置点击标志，避免重复处理
            clicked = False

        # 显示画面
        cv2.imshow(window_name, frame)

        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("程序被用户中断，正在退出...")
    except Exception as e:
        print(f"未知问题：{e}")
