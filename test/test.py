#注意：若使用到文件路径，务必使用文件的绝对路径而非相对路径
#或者在文件的开头加上：
#import os
#script_dir = os.path.dirname(os.path.abspath(__file__))
#os.chdir(script_dir)

import cv2
import time

def detailed_camera_test(path):
    print(f"\n=== 详细测试 {path} ===")
    
    # 方法1: 使用V4L2后端
    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"  使用V4L2后端无法打开 {path}")
        cap = cv2.VideoCapture(path)  # 尝试默认后端
        if not cap.isOpened():
            print(f"  使用默认后端也无法打开 {path}")
            return False
    
    print(f"  ✓ 成功打开 {path}")
    
    # 设置较低的分辨率以提高兼容性
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # 尝试读取多帧（有些摄像头需要预热）
    success_count = 0
    for i in range(10):
        ret, frame = cap.read()
        if ret:
            success_count += 1
            if i == 0:  # 第一帧成功时显示信息
                print(f"  帧尺寸: {frame.shape}")
        time.sleep(0.1)
    
    cap.release()
    
    if success_count > 0:
        print(f"  ✓ {path} 成功捕获 {success_count}/10 帧")
        return True
    else:
        print(f"  ✗ {path} 无法捕获任何帧")
        return False

# 测试所有设备
paths = ['/dev/video0', '/dev/video1', '/dev/video2', '/dev/video3']
working_cameras = []

for path in paths:
    if detailed_camera_test(path):
        working_cameras.append(path)

print(f"\n总结: 找到 {len(working_cameras)} 个可用的摄像头")
print(f"可用设备: {working_cameras}")