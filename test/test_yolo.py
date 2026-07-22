import process_lib.image_lib as lb
import cv2

# detector = lb.YOLODetector(method="rknn", model_path="/home/ubuntu/Project/run/best.rknn", num_classes=8, conf_thresh=0.5, iou_thresh=0.45, imgsz=(224, 224), cores=2)

CAMERA_WIDTH = 1280 # 1080p 1920*1080
CAMERA_HEIGHT = 720 # 1080p 1920*1080

def main():
    with lb.YOLODetector(method="rknn", model_path="/home/ubuntu/Project/Project/run/best.rknn", num_classes=2, conf_thresh=0.5, iou_thresh=0.5, imgsz=(320, 320), cores=2) as detector:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                boxes, scores, class_ids = detector.detect(frame)

                frame = detector.draw_boxes(frame, boxes, scores, class_ids)
                cv2.line(frame, (0, 180), (CAMERA_WIDTH, 180), (0, 255, 0), 2)
                cv2.line(frame, (0, 540), (CAMERA_WIDTH, 540), (0, 255, 0), 2)
                cv2.line(frame, (500, 0), (500, CAMERA_HEIGHT), (0, 0, 255), 2)
                cv2.line(frame, (900, 0), (900, CAMERA_HEIGHT), (0, 0, 255), 2)
                cv2.line(frame, (0, 360), (CAMERA_WIDTH, 360), (255, 0, 0), 2)
                cv2.line(frame, (640, 0), (640, CAMERA_HEIGHT), (255, 0, 0), 2)
                cv2.putText(frame, "y=180", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, "y=540", (10, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, "x=500", (510, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(frame, "x=900", (910, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(frame, "y=360", (10, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(frame, "x=640", (650, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)


                cv2.imshow("YOLO Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            print("Interrupted by user")
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()