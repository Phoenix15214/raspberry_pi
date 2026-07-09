import process_lib.control_lib as ctrl
import cv2
from threading import Thread
from multiprocessing import Process
import time

frame_share = ctrl.MemoryShare(name='shared_frame', shape=(480,640,3), dtype='uint8')

def Capture_Process(shm_name):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            continue
        frame_share.write(frame)

def Display_Process(shm_name):
    while True:
        frame = frame_share.read()
        if frame is not None:
            cv2.imshow("Shared Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        time.sleep(0.01)  # Add a small delay to reduce CPU usage
    cv2.destroyAllWindows()

def main():
    try:
        capture_process = Process(target=Capture_Process, args=(frame_share.name,))
        display_process = Process(target=Display_Process, args=(frame_share.name,))

        capture_process.start()
        display_process.start()

        capture_process.join()
        display_process.join()
    except KeyboardInterrupt:
        print("Exiting...")
        capture_process.terminate()
        display_process.terminate()
    finally:
        frame_share.close()

if __name__ == "__main__":
    main()