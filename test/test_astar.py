import heapq
import math
import random
import time
import numpy as np
import cv2
from scipy.interpolate import splprep, splev
from typing import List, Tuple, Optional


# ==================== 1. A* 算法（欧几里得距离 + 8方向移动） ====================
class AStarEuclidean:
    def __init__(self, grid: List[List[int]]):
        """
        grid: 二维列表，0=可通行，1=障碍物
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """
        欧几里得距离启发式
        """
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def get_neighbors(self, node: Tuple[int, int]) -> List[Tuple[Tuple[int, int], float]]:
        """
        获取8个方向的邻居，并返回 (邻居坐标, 移动代价)
        对角线移动代价为 sqrt(2)，上下左右为 1
        同时检查对角线移动时是否“穿墙”（防止斜穿障碍物角落）
        """
        x, y = node
        neighbors = []
        # 8个方向: (dx, dy)
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),          # 上下左右
            (-1, -1), (-1, 1), (1, -1), (1, 1)         # 对角线
        ]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            # 检查边界
            if not (0 <= nx < self.rows and 0 <= ny < self.cols):
                continue
            # 检查障碍物
            if self.grid[nx][ny] == 1:
                continue

            # 若为对角线移动，检查相邻两个正交格子是否都是障碍物（防止穿墙）
            if dx != 0 and dy != 0:
                # 如果水平或垂直相邻格子是障碍物，则不允许斜穿
                if self.grid[x][ny] == 1 or self.grid[nx][y] == 1:
                    continue
                step_cost = math.sqrt(2)
            else:
                step_cost = 1.0

            neighbors.append(((nx, ny), step_cost))
        return neighbors

    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        A* 主搜索
        """
        # 开放列表：元素为 (f, g, 节点)
        open_list = []
        heapq.heappush(open_list, (0.0, 0.0, start))

        parent = {start: None}
        g_cost = {start: 0.0}
        closed_set = set()

        while open_list:
            _, current_g, current = heapq.heappop(open_list)

            if current in closed_set:
                continue
            closed_set.add(current)

            # 到达目标
            if current == goal:
                path = []
                while current is not None:
                    path.append(current)
                    current = parent[current]
                path.reverse()
                return path

            # 扩展邻居
            for neighbor, step_cost in self.get_neighbors(current):
                if neighbor in closed_set:
                    continue

                tentative_g = g_cost[current] + step_cost

                if neighbor not in g_cost or tentative_g < g_cost[neighbor]:
                    parent[neighbor] = current
                    g_cost[neighbor] = tentative_g
                    f_val = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_list, (f_val, tentative_g, neighbor))

        return None  # 无解


# ==================== 2. 路径平滑（B样条） ====================
def smooth_path_bspline(path: List[Tuple[int, int]], num_interp: int = 400) -> np.ndarray:
    """
    使用 B 样条对路径进行平滑
    返回形状为 (N, 2) 的 numpy 数组，每行为 (行坐标, 列坐标)
    """
    if len(path) < 4:
        print("警告：路径点数少于4，无法进行B样条平滑，返回原始点")
        return np.array(path, dtype=np.float32)

    path_np = np.array(path, dtype=np.float32)
    # splprep 要求传入 x 和 y 序列（注意：splprep 需要 x 和 y 分开）
    # 列坐标作为 x，行坐标作为 y（OpenCV 图像坐标系）
    x_vals = path_np[:, 1]   # 列
    y_vals = path_np[:, 0]   # 行

    try:
        # k=3 三次样条，s=0 强制插值所有点（曲线经过所有原始点，但拐角处变圆）
        tck, u = splprep([x_vals, y_vals], s=0, k=3)
        u_new = np.linspace(0, 1, num_interp)
        x_smooth, y_smooth = splev(u_new, tck)
        # 恢复为 (行, 列) 格式
        smoothed = np.vstack([y_smooth, x_smooth]).T
        return smoothed
    except Exception as e:
        print(f"B样条平滑失败: {e}，返回原始点")
        return path_np


# ==================== 3. OpenCV 可视化 ====================
def visualize(
    grid: List[List[int]],
    raw_path: List[Tuple[int, int]],
    smooth_path: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    cell_size: int = 12,
    window_name: str = "A* with Euclidean Distance + Smoothing"
):
    rows, cols = len(grid), len(grid[0])
    height, width = rows * cell_size, cols * cell_size

    # 白色背景
    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    # 绘制网格线（浅灰）
    for i in range(rows + 1):
        cv2.line(img, (0, i * cell_size), (width, i * cell_size), (200, 200, 200), 1)
    for j in range(cols + 1):
        cv2.line(img, (j * cell_size, 0), (j * cell_size, height), (200, 200, 200), 1)

    # 绘制障碍物（黑色）
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 1:
                cv2.rectangle(img, (j * cell_size, i * cell_size),
                              ((j + 1) * cell_size, (i + 1) * cell_size), (0, 0, 0), -1)

    def grid_to_pixel(row, col):
        return (int(col * cell_size + cell_size / 2),
                int(row * cell_size + cell_size / 2))

    # ---- 绘制原始 A* 路径（蓝色折线 + 点） ----
    if raw_path:
        pts = [grid_to_pixel(r, c) for r, c in raw_path]
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], (255, 0, 0), 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(img, p, 2, (255, 0, 0), -1)

    # ---- 绘制平滑曲线（红色粗线） ----
    if len(smooth_path) > 0:
        pts_smooth = []
        for r, c in smooth_path:
            # 浮点坐标映射到像素（直接连续映射，不取整网格中心，保证曲线流畅）
            px = float(c) * cell_size + cell_size / 2.0
            py = float(r) * cell_size + cell_size / 2.0
            pts_smooth.append((int(px), int(py)))
        pts_array = np.array(pts_smooth, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts_array], False, (0, 0, 255), 3, cv2.LINE_AA)

    # ---- 起点（绿）和终点（品红） ----
    start_px = grid_to_pixel(start[0], start[1])
    goal_px = grid_to_pixel(goal[0], goal[1])
    cv2.circle(img, start_px, 8, (0, 255, 0), -1)
    cv2.circle(img, goal_px, 8, (255, 0, 255), -1)

    # 图例
    cv2.putText(img, "Blue: A* raw path (8-dir, Euclidean)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(img, "Red: B-spline smoothed", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 显示
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ==================== 4. 主实验程序 ====================
def generate_random_grid(size: int, obstacle_ratio: float = 0.25, seed: int = 42) -> List[List[int]]:
    """生成随机障碍物网格，保证起点 (0,0) 和终点 (size-1, size-1) 为空"""
    random.seed(seed)
    np.random.seed(seed)
    grid = np.zeros((size, size), dtype=int)
    # 随机放置障碍物
    for i in range(size):
        for j in range(size):
            if (i == 0 and j == 0) or (i == size-1 and j == size-1):
                continue
            if random.random() < obstacle_ratio:
                grid[i, j] = 1
    return grid.tolist()


def main():
    # ---------- 实验参数 ----------
    GRID_SIZE = 50               # 50x50 网格
    OBSTACLE_RATIO = 0.22        # 障碍物密度
    CELL_SIZE = 12               # 每个格子像素大小（窗口为 600x600 左右）
    SMOOTH_POINTS = 500          # 平滑曲线插值点数

    print("=== A* 算法实验（欧几里得距离 + 8方向移动）===")
    print(f"网格大小: {GRID_SIZE}x{GRID_SIZE}, 障碍物密度: {OBSTACLE_RATIO:.0%}")

    # 1. 生成地图
    grid = generate_random_grid(GRID_SIZE, OBSTACLE_RATIO, seed=42)
    start = (0, 0)
    goal = (GRID_SIZE - 1, GRID_SIZE - 1)

    # 2. 运行 A*
    astar = AStarEuclidean(grid)
    start_time = time.time()
    raw_path = astar.find_path(start, goal)
    elapsed = time.time() - start_time

    if raw_path is None:
        print("❌ A* 未找到路径！请降低障碍物密度或重新生成地图。")
        # 降级：降低密度再试一次
        print("尝试降低障碍物密度至 15% ...")
        grid = generate_random_grid(GRID_SIZE, 0.15, seed=99)
        astar = AStarEuclidean(grid)
        raw_path = astar.find_path(start, goal)
        if raw_path is None:
            print("仍然无解，程序退出。")
            return

    print(f"✅ A* 找到路径！节点数: {len(raw_path)}, 耗时: {elapsed:.4f} 秒")

    # 计算路径的欧几里得长度（近似）
    path_len = 0.0
    for i in range(len(raw_path) - 1):
        r1, c1 = raw_path[i]
        r2, c2 = raw_path[i+1]
        path_len += math.hypot(r1 - r2, c1 - c2)
    print(f"原始路径几何长度: {path_len:.2f}")

    # 3. 平滑
    smoothed = smooth_path_bspline(raw_path, num_interp=SMOOTH_POINTS)
    print(f"平滑后插值点数: {len(smoothed)}")

    # 4. 可视化
    visualize(grid, raw_path, smoothed, start, goal, cell_size=CELL_SIZE)


if __name__ == "__main__":
    main()