import cv2
import threading
import queue
import time
import collections
from moment import SerialPacket
from moment import get_stop_dynamic

class CameraBuffer:
    def __init__(self, buffer_size=5, camera_index=0):
        self.buffer_size = buffer_size
        self.frame_buffer = collections.deque(maxlen=buffer_size)
        self.lock = threading.Lock()
        self.capture_thread = None
        self.processing_thread = None
        self.running = False
        
        # 打开摄像头
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not self.cap.isOpened():
            print("无法打开摄像头")
            raise Exception("Camera open failed")
    
    def start_capture(self):
        """启动图像采集线程"""
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_frames)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        print("图像采集线程已启动")
    
    def _capture_frames(self):
        """图像采集线程函数"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame_buffer.append((frame.copy(), time.time()))
            else:
                print("摄像头读取失败")
                time.sleep(0.01)
    
    def get_latest_frame(self):
        """获取最新的一帧图像"""
        with self.lock:
            if self.frame_buffer:
                return self.frame_buffer[-1]
        return None, None
    
    def get_frame_by_index(self, index=-1):
        """按索引获取帧（-1表示最新帧，-2表示上一帧，以此类推）"""
        with self.lock:
            if 0 <= abs(index) <= len(self.frame_buffer):
                return self.frame_buffer[index]
        return None, None
    
    def get_buffer_info(self):
        """获取缓冲区信息"""
        with self.lock:
            return len(self.frame_buffer), self.buffer_size
    
    def stop(self):
        """停止所有线程"""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        self.cap.release()

def get_center_point(img, min_area_threshold=40, threshold_value=51):
    """原有的中心点检测函数"""
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

def processing_worker(camera_buffer, serial_pack, stop_event):
    """图像处理工作线程"""
    count = 0
    reset = False
    fps = 0
    current_time = time.time()
    last_time = current_time
    smooth_fps = 0
    alpha = 0.1
    
    # 创建显示窗口
    cv2.namedWindow("ROI + Center")
    cv2.createTrackbar("Threshold", "ROI + Center", 80, 255, lambda x: None)
    
    while not stop_event.is_set():
        # 获取最新帧
        frame, timestamp = camera_buffer.get_latest_frame()
        
        if frame is None:
            time.sleep(0.001)  # 短暂休眠避免空转
            continue
        
        # 计算FPS
        current_time = time.time()
        fps = 1 / (current_time - last_time) if current_time != last_time else 0
        last_time = current_time
        smooth_fps = (1 - alpha) * smooth_fps + alpha * fps
        
        # 图像处理
        height, width = 480, 640
        roi = frame[height//2 - 120: height//2 + 120, width//2 - 160: width//2 + 160].copy()
        
        threshold_value = cv2.getTrackbarPos("Threshold", "ROI + Center")
        cx, cy, binary, img_gray, output, is_junction = get_center_point(roi, 40, threshold_value)
        
        # 停止标识检测
        roi_height, roi_width = roi.shape[:2]
        is_stop = get_stop_dynamic(binary, roi_height)
        
        # 计数逻辑
        if is_stop != 0:
            print(f"识别到：{is_stop}")
            count += 1
            print(count)
            if is_stop == 0:
                reset = True
        else:
            count = 0
            reset = False
        
        # 显示信息
        buffer_len, buffer_size = camera_buffer.get_buffer_info()
        cv2.putText(binary, f"FPS: {smooth_fps:.2f}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
        cv2.putText(binary, f"Buffer: {buffer_len}/{buffer_size}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
        cv2.putText(binary, f"Stop: {is_stop}", (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
        
        cv2.imshow("ROI + Center", binary)
        
        # 串口发送
        serial_pack.insert_byte(0x08)
        serial_pack.insert_two_bytes(serial_pack.num_to_bytes(0))
        serial_pack.insert_two_bytes(serial_pack.num_to_bytes(0))
        serial_pack.insert_two_bytes(serial_pack.num_to_bytes(is_stop))
        serial_pack.insert_two_bytes(serial_pack.num_to_bytes(cx + 100))
        # serial_pack.send_packet()
        
        # 检查退出键
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC退出
            stop_event.set()
            break
        elif key == ord('b'):  # 按'b'显示缓冲区信息
            print(f"缓冲区状态: {buffer_len}/{buffer_size} 帧")
            print("最新5帧时间戳:", [ts for _, ts in list(camera_buffer.frame_buffer)[-5:]])

def main():
    # 初始化摄像头缓冲区
    camera_buffer = CameraBuffer(buffer_size=5, camera_index=0)  # 60帧缓冲区
    
    # 初始化串口
    try:
        # pack = SerialPacket(port="COM1", baudrate=115200, timeout=0.1)
        pack = SerialPacket(port="/dev/ttyUSB0", baudrate=115200, timeout=0.1)
    except Exception as e:
        print(f"串口初始化失败: {e}")
        pack = None
    
    # 启动图像采集
    camera_buffer.start_capture()
    
    # 创建停止事件
    stop_event = threading.Event()
    
    try:
        # 启动处理线程
        processing_thread = threading.Thread(
            target=processing_worker, 
            args=(camera_buffer, pack, stop_event)
        )
        processing_thread.daemon = True
        processing_thread.start()
        
        print("程序已启动，按ESC退出，按'b'查看缓冲区信息")
        
        # 等待处理线程结束
        processing_thread.join()
        
    except KeyboardInterrupt:
        print("接收到中断信号")
    finally:
        print("正在停止程序...")
        stop_event.set()
        camera_buffer.stop()
        cv2.destroyAllWindows()
        print("程序已退出")

if __name__ == "__main__":
    main()