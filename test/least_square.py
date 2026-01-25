import cv2
import numpy as np
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


# 最小二乘法拟合直线
def ls(img):

    # 预处理:灰度化,二值化,取ROI
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_binary = cv2.threshold(img_gray, 60, 255, cv2.THRESH_BINARY_INV)
    height, width = img.shape[:2]
    roi_top = height // 2  # 从图像一半高度开始
    roi_bottom = height  # 到图像底部

    # 储存中心点
    center_points = []

    # 从下往上，每隔10个像素行进行一次扫描
    for y in range(roi_bottom - 1, roi_top, -10):
        # 求每一行中值为255的x坐标
        row = img_binary[y, :]
        white_pixels_x = np.where(row == 255)[0]
        # 如果找到了白色像素点
        if len(white_pixels_x) > 0:
            # 计算白色部分中心x坐标并写入center_points
            x_center = int(np.mean(white_pixels_x))
            center_points.append((x_center, y))

    # 至少要有两个点才能拟合直线
    if len(center_points) < 2:
        return 0, width // 2, img  # 如果没有找到足够点，返回默认值

    # 使用np.polyfit对直线进行拟合,次数为1
    # 获得斜率slope和截距intercept
    x_coords, y_coords = zip(*center_points)
    params = np.polyfit(y_coords, x_coords, 1)
    slope = params[0]
    intercept = params[1]

    # 计算路径中心点的平均x坐标与中间位置的偏移
    avg_x = int(np.mean(x_coords)) - width // 2

    # 计算直线的倾斜角度
    angle_rad = np.arctan(slope)  # 得到弧度
    angle_deg = np.degrees(angle_rad)  # 转换为度

    # 绘制找到的中心点（绿色）与拟合得到的直线
    vis_image = cv2.cvtColor(img_binary, cv2.COLOR_GRAY2BGR)
    for point in center_points:
        cv2.circle(vis_image, point, 5, (0, 255, 0), -1)
    y1 = roi_bottom
    x1 = int(slope * y1 + intercept)
    y2 = roi_top
    x2 = int(slope * y2 + intercept)
    cv2.line(vis_image, (x1, y1), (x2, y2), (255, 0, 0), 3)

    # 在图像上显示计算出的角度和平均x坐标
    cv2.putText(vis_image, f'Angle: {angle_deg:.2f} degrees', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(vis_image, f'Avg X: {avg_x}', (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    return avg_x, angle_deg, vis_image


if __name__ == '__main__':

    # 输入图片
    img = cv2.imread("test1.jpg")

    # 最小二乘法拟合
    avg_x, angle, result_image = ls(img)

    # 显示图片
    stack = np.hstack((img, result_image))
    cv2.imshow("img", stack)

    # 等待按键后关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 储存结果
    cv2.imwrite(os.getcwd() + '/result_ls.jpg', result_image)
