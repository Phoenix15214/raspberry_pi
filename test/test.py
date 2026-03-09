import cv2
import process_lib.image_lib as lb
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = lb.FFT_Filtering(frame, radius=30, mode=lb.FFT_HIGHPASS)
    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()