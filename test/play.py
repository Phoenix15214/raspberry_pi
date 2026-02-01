import dxcam
import cv2
import numpy as np
import time

class ScreenCaptureDXCam:
    def __init__(self, region=None, output_color="BGR"):
        """
        DXCam - 针对Windows的高性能屏幕捕捉
        
        Args:
            region: 捕捉区域 (left, top, right, bottom)
            output_color: 输出颜色格式 "BGR" 或 "RGB"
        """
        self.camera = dxcam.create(output_color=output_color)
        self.region = region
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
    def start(self, target_fps=None):
        """开始连续捕捉"""
        if target_fps:
            self.camera.start(target_fps=target_fps, region=self.region)
        else:
            self.camera.start(region=self.region)
            
    def get_latest_frame(self):
        """获取最新帧（非阻塞）"""
        frame = self.camera.get_latest_frame()
        
        # 计算FPS
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            elapsed = time.time() - self.start_time
            self.fps = self.frame_count / elapsed
            print(f"FPS: {self.fps:.1f}")
            
        return frame
    
    def stop(self):
        """停止捕捉"""
        self.camera.stop()
    
    def capture_video(self, output_file="output.mp4", duration=10, fps=60):
        """录制视频"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame = self.get_latest_frame()
        height, width = frame.shape[:2]
        
        out = cv2.VideoWriter(
            output_file, 
            fourcc, 
            fps, 
            (width, height)
        )
        
        start_time = time.time()
        while time.time() - start_time < duration:
            frame = self.get_latest_frame()
            if frame is not None:
                out.write(frame)
                
        out.release()

# 使用示例
camera = ScreenCaptureDXCam()
camera.start(target_fps=240)

while True:
    frame = camera.get_latest_frame()
    if frame is not None:
        cv2.imshow("DXCam Capture", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    

camera.stop()