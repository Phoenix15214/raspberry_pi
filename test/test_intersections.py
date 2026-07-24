import numpy as np
import cv2
from sklearn.cluster import DBSCAN

def find_dense_clusters(points, center_x, width, eps, min_samples=3):
    """
    筛选 x 在中心附近 width 范围内的点，再找出其中密度足够（eps 内点数 >= min_samples）的簇，
    返回每个簇的中心坐标（平均 x, y）。
    """
    x_min = center_x - width / 2.0
    x_max = center_x + width / 2.0
    mask = (points[:, 0] >= x_min) & (points[:, 0] <= x_max)
    filtered = points[mask]

    if len(filtered) < min_samples:
        return np.empty((0, 2)), filtered

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(filtered)
    labels = clustering.labels_

    unique_labels = np.unique(labels)
    centers = []
    for label in unique_labels:
        if label == -1:
            continue
        cluster_points = filtered[labels == label]
        center = np.mean(cluster_points, axis=0)
        centers.append(center)

    return np.array(centers), filtered


def generate_test_data(image_size=(1000, 1000), num_clusters=3, points_per_cluster=35, noise_points=200):
    """生成模拟点云：在图像中心附近产生几个高斯簇，并添加随机噪声点。"""
    np.random.seed(42)
    width, height = image_size
    center_x = width / 2.0
    center_y = height / 2.0

    points = []
    cluster_centers = [
        (center_x - 100, center_y - 50),
        (center_x + 80, center_y + 60),
        (center_x - 20, center_y + 120),
        (center_x + 150, center_y - 80),
    ]
    for cx, cy in cluster_centers[:num_clusters]:
        cluster_points = np.random.normal(loc=(cx, cy), scale=15, size=(points_per_cluster, 2))
        points.append(cluster_points)

    noise = np.random.uniform(0, [width, height], size=(noise_points, 2))
    points.append(noise)
    points = np.vstack(points)
    return points


def draw_points_on_image(image, points, color, radius=2, thickness=-1):
    """在图像上绘制一组点（用圆表示）"""
    for (x, y) in points.astype(int):
        cv2.circle(image, (x, y), radius, color, thickness)


def main():
    # 参数设置
    image_size = (1000, 1000)   # 宽, 高
    center_x = image_size[0] / 2.0
    width = 300                 # x 筛选范围 [350, 650]
    eps = 30                    # 聚类半径
    min_samples = 3             # 簇最少点数

    # 生成测试数据
    points = generate_test_data(image_size, num_clusters=3, points_per_cluster=35, noise_points=200)

    # 调用主函数
    centers, filtered_points = find_dense_clusters(points, center_x, width, eps, min_samples)

    # ---------- 用 OpenCV 绘制可视化 ----------
    # 创建白色背景图像（BGR 格式）
    img = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 255

    # 1. 绘制所有点（浅灰色）
    draw_points_on_image(img, points, color=(200, 200, 200), radius=1)

    # 2. 绘制筛选区域内的点（蓝色）
    if len(filtered_points) > 0:
        draw_points_on_image(img, filtered_points, color=(255, 0, 0), radius=2)  # BGR: 蓝色

    # 3. 绘制簇中心（红色大圆 + 十字标记）
    if len(centers) > 0:
        for (cx, cy) in centers.astype(int):
            # 红色实心圆
            cv2.circle(img, (cx, cy), 8, (0, 0, 255), -1)
            # 白色十字线
            cv2.drawMarker(img, (cx, cy), (255, 255, 255), markerType=cv2.MARKER_CROSS,
                           markerSize=10, thickness=2)

    # 4. 绘制 x 筛选边界（绿色虚线）
    x_min = int(center_x - width / 2.0)
    x_max = int(center_x + width / 2.0)
    cv2.line(img, (x_min, 0), (x_min, image_size[1]), (0, 255, 0), 1, lineType=cv2.LINE_AA)
    cv2.line(img, (x_max, 0), (x_max, image_size[1]), (0, 255, 0), 1, lineType=cv2.LINE_AA)

    # 添加文字标注
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, f"All points: {len(points)}", (10, 30), font, 0.6, (100, 100, 100), 1)
    cv2.putText(img, f"Filtered: {len(filtered_points)}", (10, 55), font, 0.6, (255, 0, 0), 1)
    cv2.putText(img, f"Clusters: {len(centers)}", (10, 80), font, 0.6, (0, 0, 255), 1)
    cv2.putText(img, "Gray: all points, Blue: x-filtered, Red: cluster centers", (10, 110), font, 0.5, (50, 50, 50), 1)

    # 显示图像
    cv2.imshow("Dense Cluster Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 控制台输出结果
    print(f"Total points: {len(points)}")
    print(f"Filtered points (within x range): {len(filtered_points)}")
    print(f"Found {len(centers)} dense cluster(s):")
    for i, c in enumerate(centers):
        print(f"  Cluster {i+1}: ({c[0]:.1f}, {c[1]:.1f})")


if __name__ == "__main__":
    main()