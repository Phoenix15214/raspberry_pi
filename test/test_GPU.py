import cv2
import numpy as np
import time

# ==========================================
# 1. 核心处理类：纯 OpenCL 加速的自适应 Sigmoid
# ==========================================
SIGMOID_KERNEL_SRC = """
__kernel void adaptive_sigmoid(
    __global const float* img,
    __global const float* T_map,
    __global const float* K_map,
    __global uchar* dst,
    int step_img, int step_T, int step_K, int step_dst,
    int width, int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    int idx_img = y * (step_img / sizeof(float)) + x;
    int idx_T   = y * (step_T   / sizeof(float)) + x;
    int idx_K   = y * (step_K   / sizeof(float)) + x;
    int idx_dst = y * step_dst + x;

    float I = img[idx_img];
    float T = T_map[idx_T];
    float K = K_map[idx_K];

    float S  = 1.0f / (1.0f + exp(-K * (I - T)));
    float S0 = 1.0f / (1.0f + exp(K * T));
    float S1 = 1.0f / (1.0f + exp(-K * (1.0f - T)));

    float num = S - S0;
    float denom = (S1 - S0) + 1e-6f;
    float res = (num / denom) * 255.0f;

    dst[idx_dst] = (uchar)clamp(res, 0.0f, 255.0f);
}
"""

class FastAdaptiveSigmoid:
    def __init__(self, grid_size=(8, 8), k_base=5.0, k_range=(1.0, 15.0)):
        self.grid_size = grid_size
        self.k_base = k_base
        self.k_min, self.k_max = k_range

        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
        else:
            print("警告: 当前环境未开启 OpenCL 支持！")

    def process(self, image_gray):
        h, w = image_gray.shape
        rows, cols = self.grid_size

        # 1. CPU 上快速做 float32 转换与归一化，然后直接包装为 UMat
        # (RK3588 CPU 在向量化转换上极快，耗时 negligible)
        img_f_cpu = image_gray.astype(np.float32) * (1.0 / 255.0)
        img_f = cv2.UMat(img_f_cpu)

        # 2. 用 GPU 区域下采样 (INTER_AREA) 替代 Python 层的双重 for 循环
        T_grid = cv2.resize(img_f, (cols, rows), interpolation=cv2.INTER_AREA)
        cv2.max(T_grid, 0.01, T_grid)
        cv2.min(T_grid, 0.99, T_grid)

        # 计算局部方差与标准差 std = sqrt(E[X^2] - (E[X])^2)
        img_sq = cv2.multiply(img_f, img_f)
        img_sq_small = cv2.resize(img_sq, (cols, rows), interpolation=cv2.INTER_AREA)
        T_sq = cv2.multiply(T_grid, T_grid)
        
        var_grid = cv2.subtract(img_sq_small, T_sq)
        cv2.max(var_grid, 0.0, var_grid) 
        std_grid = cv2.sqrt(var_grid)

        # 计算 K_grid 并限制范围
        denom_grid = cv2.add(std_grid, 0.01)
        K_grid = cv2.divide(self.k_base, denom_grid)
        cv2.threshold(K_grid, self.k_max, self.k_max, cv2.THRESH_TRUNC, K_grid)
        cv2.max(K_grid, self.k_min, K_grid)

        # 3. 将 T 和 K 的网格插值拉伸回全图尺寸
        T_map = cv2.resize(T_grid, (w, h), interpolation=cv2.INTER_LINEAR)
        K_map = cv2.resize(K_grid, (w, h), interpolation=cv2.INTER_LINEAR)

        # 4. GPU 矢量化 Sigmoid 计算链
        neg_K = cv2.multiply(K_map, -1.0)

        # S = 1 / (1 + exp(-K * (I - T)))
        diff = cv2.subtract(img_f, T_map)
        neg_K_diff = cv2.multiply(neg_K, diff)
        exp_neg_K_diff = cv2.exp(neg_K_diff)
        S = cv2.divide(1.0, cv2.add(1.0, exp_neg_K_diff))

        # S0 = 1 / (1 + exp(K * T))
        KT = cv2.multiply(K_map, T_map)
        exp_KT = cv2.exp(KT)
        S0 = cv2.divide(1.0, cv2.add(1.0, exp_KT))

        # S1 = 1 / (1 + exp(-K * (1 - T)))
        one_minus_T = cv2.subtract(1.0, T_map)
        neg_K_one_minus_T = cv2.multiply(neg_K, one_minus_T)
        exp_neg_K_one_minus_T = cv2.exp(neg_K_one_minus_T)
        S1 = cv2.divide(1.0, cv2.add(1.0, exp_neg_K_one_minus_T))

        # res = (S - S0) / (S1 - S0 + 1e-6) * 255.0
        num = cv2.subtract(S, S0)
        denom = cv2.add(cv2.subtract(S1, S0), 1e-6)
        res_u = cv2.divide(num, denom)
        res_u = cv2.multiply(res_u, 255.0)

        # 5. 使用 OpenCV 全局函数 convertScaleAbs 将 float UMat 转为 uint8 UMat
        # convertScaleAbs 会自动把数值 clamp 到 [0, 255] 并转成 uint8
        res_u8 = cv2.convertScaleAbs(res_u)

        return res_u8.get()


# ==========================================
# 2. 摄像头读取与处理循环
# ==========================================
def main():
    # 初始化摄像头。RK3588 上通常是 /dev/video0 或 /dev/video11 等
    # 可以尝试添加 cv2.CAP_V4L2 强制使用 V4L2 后端
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    
    # 设置你期望的摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("无法打开摄像头！")
        return

    # 实例化算法类（在循环外只创建一次）
    enhancer = FastAdaptiveSigmoid(grid_size=(8, 8), k_base=5.0, k_range=(1.0, 15.0))
    
    print("开始处理... 按 'q' 键退出")

    fps_count = 0
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("未能读取到画面！")
            break

        # 算法要求输入为单通道灰度图，所以需要先转换
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 核心调用：传入灰度图，返回增强后的灰度图
        enhanced_gray = enhancer.process(gray_frame)

        # 计算并显示 FPS
        fps_count += 1
        elapsed = time.time() - t_start
        if elapsed > 1.0:
            fps = fps_count / elapsed
            print(f"当前处理帧率: {fps:.1f} FPS")
            fps_count = 0
            t_start = time.time()

        # 显示画面比对
        cv2.imshow('Original', gray_frame)
        cv2.imshow('Enhanced', enhanced_gray)

        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()