import cv2
import numpy as np
import math
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


def hl(image):

    # 取ROI
    height, width, _ = image.shape
    roi_top = height // 2
    roi = image[roi_top:, 0:width]
    img_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(img_gray, 60, 255, cv2.THRESH_BINARY_INV)

    # 边缘检测与霍夫直线检测
    edges = cv2.Canny(img_binary, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)

    # 用于存储所有有效线段的参数 (x1, y1, x2, y2, slope)
    valid_lines = []

    # 创建输出图片
    output_image = image.copy()

    # 遍历所有直线
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 绘制所有直线
            cv2.line(output_image, (x2, y2), (x1, y1), (255, 0, 255), 4)
            # 计算线段的斜率,垂直直线给一个大值
            if x2 != x1:
                slope = (y2 - y1) / (x2 - x1)
            else:
                slope = 999.0
            # 过滤掉太平坦的线段
            if abs(slope) < 0.2:
                continue
            # 存储有效线段
            valid_lines.append((x1, y1, x2, y2, slope))

    if valid_lines:
        # 计算平均斜率与中心线在ROI底部的x坐标
        avg_slope = np.mean([l[4] for l in valid_lines])
        all_x = [l[0] for l in valid_lines] + [l[2] for l in valid_lines]
        p_start_x = int(np.mean(all_x))

        # 计算中心线在ROI顶部的X坐标,斜率很小时认为delta_x = 0
        delta_y = height - roi_top
        if abs(avg_slope) < 0.001:
            delta_x = 0
        else:
            delta_x = int(delta_y / avg_slope)
        p_end_x = p_start_x - delta_x

        # 使用p_start_x计算中心偏移
        offset_x = p_start_x - (width // 2)

        # 计算角度
        center_angle_rad = math.atan(avg_slope)
        angle_horiz = math.degrees(center_angle_rad)

        # 转换为与垂直方向的夹角 (垂直向前为 0 度。左转为负，右转为正)
        # 确保斜率接近垂直时角度接近 0
        if abs(avg_slope) > 999:
            center_angle = 0.0
        else:
            center_angle = (90 - angle_horiz) if angle_horiz > 0 else (90 + angle_horiz)

        # 绘制中心线及其他可视化
        cv2.line(output_image, (p_start_x, height), (p_end_x, roi_top), (0, 255, 255), 4)
        cv2.putText(output_image, f'Angle: {center_angle:.2f} degrees', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(output_image, f'Avg X: {offset_x}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    else:
        # 未检测到有效轨道
        offset_x = 0
        center_angle = 0.0

    return offset_x, center_angle, output_image


if __name__ == '__main__':
    # 读取图片
    img = cv2.imread("test1.jpg")
    # 调用循迹函数
    offset, angle, processed_img = hl(img)

    # 显示结果
    cv2.imshow('Tracking Result (Single Line Aggregation)', processed_img)
    cv2.waitKey(0)
    cv2.imwrite(os.getcwd() + '/result_hl.jpg', processed_img)
    cv2.destroyAllWindows()
