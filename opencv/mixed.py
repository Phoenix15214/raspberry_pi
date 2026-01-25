import cv2
import numpy as np
import threading
import time
from collections import deque
from moment import SerialPacket, count_red_green_pixels_rgb
from moment import recognize_text_opencv
from moment import get_stop_dynamic

class CameraThread(threading.Thread):
    def __init__(self, camera_id, camera_index, width=320, height=240, fps=15, buffer_size=2):
        threading.Thread.__init__(self)
        self.camera_id = camera_id
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_size = buffer_size
        
        self.frame = None
        self.latest_frame = None
        self.ret = False
        self.running = True
        self.lock = threading.Lock()
        self.frame_count = 0
        self.last_frame_time = time.time()
        
    def run(self):
        print(f"启动摄像头 {self.camera_id} (索引: {self.camera_index}) 线程")
        
        # 初始化摄像头
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"无法打开摄像头 {self.camera_id}")
            return
            
        # 设置摄像头参数
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # 尝试使用V4L2后端（Linux）
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        except:
            pass
            
        retry_count = 0
        max_retries = 5
        
        while self.running:
            try:
                ret, frame = cap.read()
                
                if ret:
                    with self.lock:
                        self.frame = frame.copy()
                        self.latest_frame = frame.copy()
                        self.ret = True
                        self.frame_count += 1
                        self.last_frame_time = time.time()
                    retry_count = 0  # 重置重试计数
                else:
                    print(f"摄像头 {self.camera_id} 读取失败")
                    retry_count += 1
                    if retry_count > max_retries:
                        print(f"摄像头 {self.camera_id} 多次读取失败，尝试重新初始化")
                        cap.release()
                        time.sleep(0.5)
                        cap = cv2.VideoCapture(self.camera_index)
                        if cap.isOpened():
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                            cap.set(cv2.CAP_PROP_FPS, self.fps)
                        retry_count = 0
                    
                # 控制读取频率
                time.sleep(1.0 / self.fps)
                
            except Exception as e:
                print(f"摄像头 {self.camera_id} 线程错误: {e}")
                time.sleep(0.1)
        
        # 清理
        cap.release()
        print(f"摄像头 {self.camera_id} 线程已停止")
    
    def get_frame(self):
        """获取最新帧"""
        with self.lock:
            if self.ret and self.latest_frame is not None:
                return True, self.latest_frame.copy()
            else:
                return False, None
    
    def stop(self):
        """停止线程"""
        self.running = False

def get_center_point(img, min_area_threshold=40, threshold_value=51):
    """获取道路中心点"""
    img_output = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
    
    cnts = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cx, cy = -1, -1
    is_junction = 0

    if len(cnts) > 0:
        largest_cnt = max(cnts, key=cv2.contourArea)
        m = cv2.moments(largest_cnt)
        if m['m00'] > 0:
            cx = int(m['m10'] / m['m00'])
            cy = int(m['m01'] / m['m00'])
            cv2.circle(img_output, (cx, cy), 5, (0, 255, 0), -1)
            cv2.line(img_output, (cx, 0), (cx, img_output.shape[0]), (255, 0, 0), 2)
        
        cnts_sorted = sorted(cnts, key=cv2.contourArea, reverse=True)
        main_contours = [cnt for cnt in cnts_sorted if cv2.contourArea(cnt) > min_area_threshold]
        is_junction = 1 if len(main_contours) > 2 else 0

    return cx, cy, img_binary, img_gray, img_output, is_junction

def process_image(img, threshold_value=51):
    """处理图像"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
    return binary, gray

def main():
    # 检测可用摄像头
    print("检测摄像头...")
    available_cameras = []
    for i in range(5):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                ret, frame = cap.read()
                if ret:
                    available_cameras.append(i)
                    print(f"摄像头 {i}: 可用")
                cap.release()
        except Exception as e:
            print(f"检测摄像头 {i} 时出错: {e}")
    
    print(f"可用的摄像头索引: {available_cameras}")
    
    if len(available_cameras) < 2:
        print("需要至少2个摄像头")
        return
    
    # 启动摄像头线程
    camera2 = CameraThread("Camera1", available_cameras[0], width=320, height=240, fps=15)
    camera1 = CameraThread("Camera2", available_cameras[1], width=320, height=240, fps=15)
    
    camera1.start()
    camera2.start()
    
    # 等待摄像头初始化
    time.sleep(2)
    pack = SerialPacket(port="/dev/ttyUSB0", baudrate=115200, timeout=0.1)
    # 创建显示窗口
    cv2.namedWindow("Camera1 - ROI")
    cv2.namedWindow("Camera2 - ROI")
    # cv2.namedWindow("Camera1 - Original")
    # cv2.namedWindow("Camera2 - Original")
    
    cv2.createTrackbar("Threshold1", "Camera1 - ROI", 100, 255, lambda x: None)
    cv2.createTrackbar("Threshold2", "Camera2 - ROI", 100, 255, lambda x: None)
    
    pack = SerialPacket(port="/dev/ttyUSB0", baudrate=115200, timeout=0.1)
    
    frame_count = 0
    last_status_time = time.time()
    
    try:
        while True:
            start_time = time.time()
            
            # 从线程获取帧
            ret1, frame1 = camera1.get_frame()
            ret2, frame2 = camera2.get_frame()
            
            if not ret1 or not ret2:
                print("等待摄像头数据...")
                time.sleep(0.1)
                continue
            
            # 处理帧
            height1, width1 = frame1.shape[:2]
            height2, width2 = frame2.shape[:2]
            
            # 定义ROI区域
            roi1_height, roi1_width = 120, 160
            roi2_height, roi2_width = 240, 320
            
            roi1 = frame1[height1//2 - roi1_height//2: height1//2 + roi1_height//2, 
                          width1//2 - roi1_width//2: width1//2 + roi1_width//2].copy()
            roi2 = frame2[height2//2 - roi2_height//2: height2//2 + roi2_height//2, 
                          width2//2 - roi2_width//2: width2//2 + roi2_width//2].copy()
            
            # 获取阈值
            threshold_value1 = cv2.getTrackbarPos("Threshold1", "Camera1 - ROI")
            threshold_value2 = cv2.getTrackbarPos("Threshold2", "Camera2 - ROI")
            
            # 图像处理
            try:
                # 摄像头1处理（红绿灯和字符识别）
                red_count, green_count, yellow_count= count_red_green_pixels_rgb(roi1)
                rgb_control = 1 if red_count > 2000 else (2 if green_count > 1000 else(3 if yellow_count > 3000 else 0) )
                
                binary1, gray = process_image(roi1, threshold_value1)
                text = recognize_text_opencv(gray)
                str_control = 1 if text == 'L' else (2 if text == 'R' else 0)
                
                # 摄像头2处理（道路中心点）
                cx, cy, binary2, img_gray, output, is_junction = get_center_point(roi2, 40, threshold_value2)
                roi_height, roi_width = roi2.shape[:2]
                is_stop = get_stop_dynamic(binary2, roi_height)
                print(f"红绿灯：{rgb_control}, 字符:{str_control}, 停止:{is_stop}, 路口:{is_junction}")
                # print(cx)
                # 定期打印状态
                # current_time = time.time()
                # if current_time - last_status_time > 2.0:  # 每2秒打印一次
                #     print(f"红绿灯：{rgb_control}, 字符:{str_control}, 停止:{is_stop}, 路口:{is_junction}")
                #     # print(f"摄像头1帧率: {camera1.frame_count / (current_time - camera1.last_frame_time + 1e-5):.1f} fps")
                #     # print(f"摄像头2帧率: {camera2.frame_count / (current_time - camera2.last_frame_time + 1e-5):.1f} fps")
                #     last_status_time = current_time
                    
            except Exception as e:
                print(f"图像处理错误: {e}")
                continue
            
            # 显示结果
            # cv2.imshow("Camera1 - Original", frame1)
            # cv2.imshow("Camera2 - Original", frame2)
            cv2.imshow("Camera1 - ROI", roi1)
            cv2.imshow("Camera2 - ROI", binary2)
            
            # 发送数据包（如果需要）
            pack.insert_byte(0x08)
            pack.insert_two_bytes(pack.num_to_bytes(rgb_control))
            pack.insert_two_bytes(pack.num_to_bytes(str_control))
            pack.insert_two_bytes(pack.num_to_bytes(is_stop))
            pack.insert_two_bytes(pack.num_to_bytes(cx + 100))
            # pack.send_packet()
            
            # frame_count += 1
            
            # # 计算并显示处理帧率
            # processing_time = time.time() - start_time
            # processing_fps = 1.0 / processing_time if processing_time > 0 else 0
            # if frame_count % 30 == 0:
            #     print(f"处理帧率: {processing_fps:.1f} fps")
            
            # 退出检测
            if cv2.waitKey(1) & 0xFF == 27:
                break
                
    except KeyboardInterrupt:
        print("程序中断")
    except Exception as e:
        print(f"主循环错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 停止摄像头线程
        camera1.stop()
        camera2.stop()
        camera1.join()
        camera2.join()
        cv2.destroyAllWindows()
        print("程序已退出")

if __name__ == "__main__":
    main()