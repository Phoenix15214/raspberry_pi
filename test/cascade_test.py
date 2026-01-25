import cv2
import time

# 帧率显示部分变量
fps = 0
current_time = time.time()
last_time = current_time
smooth_fps = 0
alpha = 0.1


cap = cv2.VideoCapture(0)
# 使用HAAR级联分类器进行人脸检测
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
print(face_cascade)
while True:
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        # 打印出识别到的人脸位置并在图像上绘制矩形框
        for (x, y, w, h) in faces:
            print(x, y, w, h)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # 计算并显示帧率
        current_time = time.time()
        fps = 1 / (current_time - last_time)
        last_time = current_time
        smooth_fps = (1 - alpha) * smooth_fps + alpha * fps
        cv2.putText(frame, f"FPS: {smooth_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Camera Feed", frame)
        # cv2.imshow("gray", gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    
cap.release()
cv2.destroyAllWindows()