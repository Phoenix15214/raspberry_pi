import cv2
import numpy as np
CAMERA_FPS = 30
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
FRAME_CENTER_X = CAMERA_WIDTH // 2
FRAME_CENTER_Y = CAMERA_HEIGHT // 2

white = np.full((CAMERA_HEIGHT, CAMERA_WIDTH), 255, dtype=np.uint8)

def open_camera():
    try:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        cap.set(cv2.CAP_PROP_EXPOSURE, 30)
        actual_auto_exp = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        actual_exp = cap.get(cv2.CAP_PROP_EXPOSURE)
        print(f"Camera settings: Auto Exposure={actual_auto_exp}, Exposure={actual_exp}")
        return cap
    except Exception as e:
        print(f"Error opening camera: {e}")
        raise RuntimeError("Failed to open camera.")
        return None

def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    pink_mask = cv2.inRange(frame, (130, 20, 100), (180, 150, 255))
    black_mask = cv2.inRange(frame, (0, 0, 0), (180, 255, 100))
    white_mask = cv2.inRange(frame, (20, 0, 170), (160, 50, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    pink_mask = cv2.morphologyEx(pink_mask, cv2.MORPH_CLOSE, kernel)
    black_mask = cv2.dilate(black_mask, kernel, iterations=2)
    # black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    pink_frame = cv2.bitwise_and(white, white, mask=pink_mask)
    black_frame = cv2.bitwise_and(white, white, mask=black_mask)
    white_frame = cv2.bitwise_and(white, white, mask=white_mask)

    return pink_frame, black_frame, white_frame

def main():
    cap = open_camera()
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pink_frame, black_frame, white_frame = preprocess_frame(frame)

        cv2.imshow('Pink Frame', pink_frame)
        cv2.imshow('Original Frame', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()