import cv2
from ultralytics import YOLO
import time
import numpy as np  # 【修改1】移到顶部，修复np未定义错误

# 全局变量初始化
last_time = time.time()
smooth_fps = 0.0
alpha = 0.1

def main():
    global last_time, smooth_fps  # 【修改2】添加global声明，修复变量作用域
    
    print("YOLO11n with ONNX format")
    print("=" * 40)
    
    # 1. 加载ONNX模型
    onnx_path = "/home/pi/rasp_projects/project/HUST_STI_1/Devices/Vision/yolo/11n_02.onnx"
    
    try:
        model = YOLO(onnx_path, task='detect')
        print("✅ ONNX model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load ONNX model: {e}")
        return
    
    # 2. 打开摄像头
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Failed to open camera")
        return
    
    print("Camera opened. Starting detection...")
    print("Press 'q' to quit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            # 使用ONNX推理
            results = model.predict(
                source=frame,
                imgsz=256,      # 使用更小的推理尺寸加速
                conf=0.6,
                verbose=False,
                agnostic_nms=False,
                max_det=20,
                stream = True
            )
            
            # 计算FPS
            current_time = time.time()
            fps_raw = 1.0 / (current_time - last_time)  # 【修改3】重命名临时变量
            last_time = current_time
            smooth_fps = (alpha * fps_raw) + (1 - alpha) * smooth_fps
            
            # 处理结果（保持原有结构不变）
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes_data = result.boxes.data.cpu().numpy()
                    
                    if not np.isnan(boxes_data).any():
                        annotated = result.plot()
                        cv2.putText(annotated, f"FPS: {smooth_fps:.1f}", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.imshow("YOLO11n ONNX", annotated)
                    else:
                        cv2.putText(frame, f"FPS: {smooth_fps:.1f}", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.imshow("YOLO11n ONNX", frame)  # 保持原逻辑显示frame
                else:
                    cv2.putText(frame, f"FPS: {smooth_fps:.1f}", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("YOLO11n ONNX", frame)
            
            # 按q退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()