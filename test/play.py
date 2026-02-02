import dxcam
import cv2
import pydirectinput as pdi
import win32api
import numpy as np
import time
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import process_lib.control_lib as cb
import process_lib.image_lib as lb

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
        """获取最新帧"""
        frame = self.camera.grab()
        
        # 计算FPS
        self.frame_count += 1
        if self.frame_count % 1 == 0:
            elapsed = time.time() - self.start_time
            self.fps = self.frame_count / elapsed
            # print(f"FPS: {self.fps:.1f}")
            
        return frame
    
    def stop(self):
        """停止捕捉"""
        self.camera.stop()

MIN_CONTOUR_AREA = 200
CENTER_WIDTH = 1280 / 5
CENTER_HEIGHT = 800 / 5
RUN_TIME = 15

# 使用示例
camera = ScreenCaptureDXCam()
camera.start(target_fps=60)
pidx = cb.PID_Inc(0.8, 0.005, 0.0, CENTER_WIDTH,2,integral_limit=100,min_output=1)
pidy = cb.PID_Inc(0.8, 0.005, 0.0, CENTER_HEIGHT,2,integral_limit=100,min_output=1)
accumulatex = 0.0
accumulatey = 0.0

start_time = time.time()

while True:
    frame = camera.get_latest_frame()
    if frame is not None:
        frame = cv2.resize(frame, None, fx=0.2, fy=0.2)
        blue = lb.color_extraction_dynamic(frame, np.array([78, 43, 46]), np.array([99, 255, 255]))
        gray = cv2.cvtColor(blue, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(gray, 114, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        valid_contour_centers = []
        for contour in contours:
            area = abs(cv2.contourArea(contour, True))
            if area > MIN_CONTOUR_AREA:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                circle_area = np.pi * radius * radius
                if area != 0 and abs((area - circle_area)) / area < 0.2:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        valid_contour_centers.append([cx, cy])
                        # print(f"x坐标:{cx}, y坐标:{cy},面积:{area}")
        if valid_contour_centers:
            valid_contour_centers = np.asarray(valid_contour_centers)
            target = valid_contour_centers[cb.Get_Closest_Target([CENTER_WIDTH, CENTER_HEIGHT], valid_contour_centers)]
            pidx.frequency = camera.fps
            pidy.frequency = camera.fps
            movex = -pidx.Cal_PID(target[0]) * 5
            movey = -pidy.Cal_PID(target[1]) * 5
            accumulatex += movex
            accumulatey += movey
            start = time.time()
            pdi.moveRel(int(accumulatex), int(accumulatey), 0.1)
            pdi.press('h')
            print(time.time() - start)
        else:
            pass
        binary = cv2.resize(binary, None, fx=0.4, fy=0.4)
        cv2.imshow("DXCam Capture", binary)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        elif time.time() - start_time > RUN_TIME:
            break
    

camera.stop()
cv2.destroyAllWindows()