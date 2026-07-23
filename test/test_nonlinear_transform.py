import cv2
import numpy as np

def sigmoid_curve(image_gray, k=5.0, T=0.5):
    img_float = image_gray.astype(np.float32) / 255.0
    S = 1.0 / (1.0 + np.exp(-k * (img_float - T)))
    S0 = 1.0 / (1.0 + np.exp(k * T))
    S1 = 1.0 / (1.0 + np.exp(-k * (1.0 - T)))
    transformed = (S - S0) / (S1 - S0)
    transformed = np.clip(transformed, 0.0, 1.0)
    return (transformed * 255).astype(np.uint8)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    k = 5.0
    T = 0.5   # 初始拐点对应 127.5
    step_k = 0.5
    step_T = 0.02

    print("控制说明：")
    print("  [A] 减小 k   [D] 增大 k")
    print("  [W] 增大阈值 T（更白才能变白）")
    print("  [S] 减小阈值 T（更灰就能变白）")
    print("  [R] 重置 k=5.0, T=0.5")
    print("  [Q] 退出")
    print(f"当前 k={k:.1f}, T={T:.2f} (对应灰度 {T*255:.0f})")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        enhanced = sigmoid_curve(gray, k, T)

        # 显示参数
        info = f"k={k:.1f}  T={T:.2f} ({int(T*255)})"
        cv2.putText(enhanced, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        combined = np.hstack((gray, enhanced))
        cv2.imshow('Original (L)  vs  S-Curve with tunable threshold (R)', combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a') or key == ord('A'):
            k = max(0.5, k - step_k)
            print(f"k={k:.1f}, T={T:.2f} ({int(T*255)})")
        elif key == ord('d') or key == ord('D'):
            k = min(12.0, k + step_k)
            print(f"k={k:.1f}, T={T:.2f} ({int(T*255)})")
        elif key == ord('w') or key == ord('W'):
            T = min(0.95, T + step_T)
            print(f"k={k:.1f}, T={T:.2f} ({int(T*255)})")
        elif key == ord('s') or key == ord('S'):
            T = max(0.05, T - step_T)
            print(f"k={k:.1f}, T={T:.2f} ({int(T*255)})")
        elif key == ord('r') or key == ord('R'):
            k, T = 5.0, 0.5
            print("重置参数")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()