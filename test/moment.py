import cv2


def get_center_point(img):

    img_output = img.copy()

    # 取ROI

    # roi = img[:height//3, :]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_binary = cv2.threshold(img_gray, 60, 255, cv2.THRESH_BINARY_INV)

    # 提取最大轮廓
    cnts = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    cx, cy = -1, -1

    # 存在轮廓时
    if len(cnts) > 0:
        # 提取最大面积轮廓
        largest_cnt = max(cnts, key=cv2.contourArea)
        # 使用cv2.moments计算图形矩
        m = cv2.moments(largest_cnt)
        # 计算中心坐标
        # 防止除0
        if m['m00'] > 0:
            cx = int(m['m10'] / m['m00'])
            cy = int(m['m01'] / m['m00'])
            # 绘制中心点
            cv2.circle(img_output, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(img_output, f'Avg X: {cx}', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(img_output, f'contour: {len(cnts)}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            for cnt in cnts:
                cv2.drawContours(img_output, cnt, -1, (0, 255, 0), 3)
            # 绘制轮廓
    return cx, cy, img_output


if __name__ == '__main__':
    # 读取图片
    img = cv2.imread("test1.jpg")
    # 取ROI
    height, width = img.shape[:2]
    roi_top = height // 2
    roi = img[roi_top:, :]
    # 计算中心点坐标
    roi = cv2.imread("roi.jpg")
    x, y, output = get_center_point(roi)
    # 显示图片
    cv2.imshow("img", output)
    cv2.imshow("cut", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


