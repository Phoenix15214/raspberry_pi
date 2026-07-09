import process_lib.image_lib as lb
import cv2

# detector = lb.YOLODetector(method="rknn", model_path="/home/ubuntu/Project/run/best.rknn", num_classes=8, conf_thresh=0.5, iou_thresh=0.45, imgsz=(224, 224), cores=2)


def main():
    with lb.YOLODetector(method="rknn", model_path="/home/ubuntu/Project/run/best.rknn", num_classes=8, conf_thresh=0.5, iou_thresh=0.45, imgsz=(640, 640), cores=2) as detector:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, 30)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break

                boxes, scores, class_ids = detector.detect(frame)

                frame = detector.draw_boxes(frame, boxes, scores, class_ids)

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