import cv2
import numpy as np
import onnxruntime as ort
import time

def parse_yolo_custom(output, conf_threshold=0.4, iou_threshold=0.45, input_shape=(224, 224), num_classes=8):
    """
    专为输出形状为 [1, 12, 1029] 且 nc=8 的模型设计的新解析函数。
    假设：12 = 4(bbox坐标xywh) + 8(类别分数)
    注意：此格式没有单独的‘物体置信度’，因此置信度 = 各类别分数中的最大值。
    """
    predictions = np.squeeze(output, 0)  # [12, 1029]
    predictions = predictions.T          # [1029, 12]

    # 1. 分离数据：前4列为坐标，后8列为类别分数
    boxes = predictions[:, :4]          # xywh
    class_scores = predictions[:, 4:4+num_classes]  # 8个类别分数

    # 2. 置信度 = 8个类别分数中的最大值 (因为无单独物体置信度)
    scores = np.max(class_scores, axis=1)
    class_ids = np.argmax(class_scores, axis=1)

    # 3. 调试：打印原始分数范围，验证数据
    # print(f"原始分数范围: min={class_scores.min():.4f}, max={class_scores.max():.4f}, mean={class_scores.mean():.6f}")

    # 4. 根据置信度阈值筛选
    valid_indices = scores > conf_threshold
    boxes = boxes[valid_indices]
    scores = scores[valid_indices]
    class_ids = class_ids[valid_indices]

    if len(boxes) == 0:
        # print(f"警告: 经过置信度阈值 {conf_threshold} 筛选后，无有效检测框。请尝试降低阈值。")
        # 调试：即使没框，也输出一些原始数据看看
        # print(f"所有预测框中，最大的类别分数为: {np.max(class_scores):.4f}")
        return np.array([]), np.array([]), np.array([])

    # print(f"初步筛选后保留 {len(boxes)} 个框")

    # 5. 坐标转换与NMS (与之前相同，但注意坐标值可能未归一化)
    # 首先，检查坐标值范围。如果模型输出未做sigmoid，中心坐标可能不在[0,1]。
    # 尝试对坐标应用sigmoid激活（常见于YOLO原始输出）
    # import scipy.special
    # boxes[:, :2] = scipy.special.expit(boxes[:, :2])  # 对x_center, y_center做sigmoid
    # boxes[:, 2:] = scipy.special.expit(boxes[:, 2:])  # 对width, height做sigmoid

    x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    # 如果坐标已经是相对值且在合理范围，直接转换
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    # 6. NMS
    keep_indices = nms(boxes_xyxy, scores, iou_threshold)
    boxes = boxes[keep_indices]
    scores = scores[keep_indices]
    class_ids = class_ids[keep_indices]
    # print(f"经NMS后保留 {len(boxes)} 个框")

    # 7. 将归一化坐标转换为像素坐标 (假设坐标已归一化)
    img_h, img_w = input_shape
    boxes[:, 0] *= img_w  # x_center
    boxes[:, 1] *= img_h  # y_center
    boxes[:, 2] *= img_w  # width
    boxes[:, 3] *= img_h  # height

    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    boxes_pixel = np.stack([x1, y1, x2, y2], axis=1).astype(int)

    return boxes_pixel, scores, class_ids

def draw_detections_with_label(image, boxes, scores, class_ids, class_names=None):
    """
    在图像上绘制检测框，并在框的左上角添加标签（类别: 置信度）。
    - image: 要绘制的原始图像 (OpenCV格式，BGR)
    - boxes: 边界框数组，格式为 [x1, y1, x2, y2]（像素坐标）
    - scores: 置信度数组
    - class_ids: 类别ID数组
    - class_names: (可选) 类别名称列表。如果为None，则使用数字ID。
    """
    # 如果未提供类别名称，则使用简单的数字字符串 ['0','1',...]
    if class_names is None:
        # 根据你的类别数8，生成 ['0', '1', ..., '7']
        class_names = [str(i) for i in range(8)]

    for i, (box, score, cls_id) in enumerate(zip(boxes, scores, class_ids)):
        x1, y1, x2, y2 = box

        # 1. 画矩形框 (绿色，线宽2)
        color = (0, 255, 0)  # BGR: 绿色
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # 2. 准备标签文本
        # 将类别ID转换为名称（确保不越界）
        if 0 <= cls_id < len(class_names):
            cls_name = class_names[cls_id]
        else:
            cls_name = str(cls_id)
        label = f"{cls_name}: {score:.2f}"

        # 3. 计算标签文本的大小
        (label_width, label_height), baseline = cv2.getTextSize(
            label, 
            cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=0.5,  # 字体大小
            thickness=1     # 字体粗细
        )

        # 4. 计算标签背景框的位置 (在检测框左上角上方)
        label_y1 = max(y1 - label_height - baseline, 0)  # 确保不超出图像顶部
        label_y2 = y1
        label_x1 = x1
        label_x2 = x1 + label_width

        # 5. 绘制半透明的标签背景 (深灰色)
        sub_img = image[label_y1:label_y2, label_x1:label_x2]
        if sub_img.size > 0:
            # 创建一个与背景区域同大小的深色矩形
            white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 40  # 深灰色 (BGR: 40,40,40)
            # 将原图背景与深色矩形混合 (这里使用简单覆盖，如需半透明效果需使用cv2.addWeighted)
            image[label_y1:label_y2, label_x1:label_x2] = white_rect

        # 6. 绘制标签文字 (白色)
        text_color = (255, 255, 255)  # BGR: 白色
        cv2.putText(image, 
                    label, 
                    (x1, y1 - baseline),  # 文字基线位置
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5,  # 字体大小
                    text_color, 
                    thickness=1,
                    lineType=cv2.LINE_AA)
    return image

def nms(boxes, scores, iou_threshold):
    """非极大值抑制 (与之前相同)"""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


def main():
    # --- 1. 创建优化的ONNX Runtime会话 ---
    model_path = "/home/pi/rasp_projects/project/HUST_STI_1/Devices/Vision/yolo/11n_02_int8_simple.onnx"  # 或你的onnx模型路径
    providers = ['CPUExecutionProvider']
    
    # 关键优化配置
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4   # 使用4个线程进行并行计算
    options.inter_op_num_threads = 4
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # 可选：启用动态量化以获得额外加速（可能轻微影响精度）
    # options.optimized_model_filepath = "model_quantized.onnx"
    # options.add_session_config_entry("session.intra_op.allow_spinning", "1") 
    
    print(f"[1] Loading model with ONNX Runtime...")
    session = ort.InferenceSession(model_path, sess_options=options, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_shape = session.get_inputs()[0].shape  # [1, 3, H, W]
    print(f"    Input shape: {input_shape}")
    
    # --- 2. 初始化摄像头 ---
    print("[2] Initializing camera...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("    Failed to open camera.")
        return
    
    print("[3] Starting inference (Press 'q' to quit)...")
    frame_count = 0
    start_time = time.time()
    
    # 预热
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    _ = session.run([output_name], {input_name: dummy_input})
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # --- 3. 预处理 ---
            # 调整尺寸并转换为模型输入格式 [1, 3, H, W]
            h, w = input_shape[2], input_shape[3]
            img_resized = cv2.resize(frame, (w, h))
            img_data = img_resized.transpose(2, 0, 1)  # HWC to CHW
            img_data = np.expand_dims(img_data, 0).astype(np.float32) / 255.0  # 添加批次维度并归一化
            
            # --- 4. 推理 ---
            infer_start = time.time()
            outputs = session.run([output_name], {input_name: img_data})
            infer_time = time.time() - infer_start
            
            # --- 5. 后处理 (需要你根据模型输出调整) ---
            # 这里 outputs[0] 的形状取决于你的模型，例如 [1, 84, 8400]
            # 你需要在这里解析边界框、置信度和类别
            # 示例：boxes, scores, class_ids = parse_yolo_output(outputs[0])
            #        frame = draw_detections(frame, boxes, scores, class_ids)
            boxes, scores, class_ids = parse_yolo_custom(outputs[0],
                                                         conf_threshold=0.4,
                                                         input_shape=(h, w),
                                                         num_classes=8)
            # print(f"解析到 {len(boxes)} 个候选框")
            if len(boxes) > 0:
                # print(f"第一个框坐标 (x1, y1, x2, y2): {boxes[0]}")
                # print(f"对应的置信度: {scores[0]}")
                # print(f"对应的类别ID: {class_ids[0]}")
                custom_names = ['1', '2', '3', '4', '5', '6', '7', '8']
                orig_h, orig_w = frame.shape[:2]  # 原始图像高宽 (例如 480, 640)
                model_h, model_w = h, w           # 模型输入高宽 (224, 224)
                scale_x = orig_w / model_w
                scale_y = orig_h / model_h
                scaled_boxes = []
                for box in boxes:
                    x1, y1, x2, y2 = box
                    # 将坐标按比例缩放
                    x1_scaled = int(x1 * scale_x)
                    y1_scaled = int(y1 * scale_y)
                    x2_scaled = int(x2 * scale_x)
                    y2_scaled = int(y2 * scale_y)
                    scaled_boxes.append([x1_scaled, y1_scaled, x2_scaled, y2_scaled])
                class_names = ['1', '2', '3', '4', '5', '6', '7', '8']
                frame = draw_detections_with_label(frame, scaled_boxes, scores, class_ids, class_names=class_names)
                # for box in boxes:
                #     x1, y1, x2, y2 = box
                #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 临时：显示帧率和推理时间
            fps = 1.0 / infer_time if infer_time > 0 else 0
            cv2.putText(frame, f"ORT FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Infer: {infer_time*1000:.1f}ms", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("YOLO11n - ONNX Runtime", frame)
            frame_count += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        avg_fps = frame_count / (time.time() - start_time)
        print(f"\n[Info] Average FPS: {avg_fps:.1f}")

if __name__ == "__main__":
    main()