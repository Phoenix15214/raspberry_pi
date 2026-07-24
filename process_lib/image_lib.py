import cv2
import numpy as np
import time

# 傅里叶变换有关参数
FFT_HIGHPASS = 0
FFT_LOWPASS = 1

# 颜色提取相关参数
RED, GREEN, BLUE, WHITE, YELLOW, ORANGE, PURPLE, PINK, BROWN, GRAY, BLACK = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10

RED_LOWER1 = np.array([0, 43, 46])
RED_UPPER1 = np.array([10, 255, 255])
RED_LOWER2 = np.array([156, 43, 46])
RED_UPPER2 = np.array([180, 255, 255])

WHITE_LOWER = np.array([0, 0, 221])
WHITE_UPPER = np.array([180, 30, 255])

GREEN_LOWER = np.array([35, 43, 46])
GREEN_UPPER = np.array([77, 255, 255])

YELLOW_LOWER = np.array([26, 43, 46])
YELLOW_UPPER = np.array([34, 255, 255])

BLUE_LOWER = np.array([100, 43, 46])
BLUE_UPPER = np.array([124, 255, 255])

ORANGE_LOWER = np.array([11, 43, 46])
ORANGE_UPPER = np.array([25, 255, 255])

PURPLE_LOWER = np.array([125, 43, 46])
PURPLE_UPPER = np.array([155, 255, 255])

PINK_LOWER = np.array([160, 50, 100])
PINK_UPPER = np.array([180, 150, 220])

BROWN_LOWER = np.array([10, 100, 50])
BROWN_UPPER = np.array([20, 200, 150])

GRAY_LOWER = np.array([0, 0, 50])
GRAY_UPPER = np.array([180, 50, 200])

BLACK_LOWER = np.array([0, 0, 0])
BLACK_UPPER = np.array([180, 150, 50])

# 中心点计算相关参数
CENTER_ALL = 0
CENTER_MAX = 1
CENTER_SINGLE = 2





# 傅里叶变换有关函数
def FFT_Filtering(img, radius=30, mode=FFT_HIGHPASS):
    """
    对输入图像进行高通滤波处理，增强图像的边缘和细节。默认高通滤波。
    参数:
        img: 输入图像，灰度图与彩色图均可。
        radius: 高通滤波器的半径，默认为30。
    返回:
        处理后的图像。
    """
    if img is None:
        raise ValueError("输入图像不能为空")
    if img.ndim not in [2, 3]:
        raise ValueError("输入必须为彩色图或灰度图")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    
    rows, cols = img.shape
    crow, ccol = rows // 2 , cols // 2
    
    if(radius < 0):
        raise ValueError("半径必须为正整数")
    elif radius == 0:
        radius = 1
    elif(radius > min(crow, ccol)):
        raise ValueError("半径过大，无法应用滤波器")
    
    img_float32 = np.float32(img)
    
    dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    if mode == FFT_LOWPASS:
        mask = np.zeros((rows, cols, 2), np.uint8)
        cv2.circle(mask, (ccol, crow), radius, (1, 1, 1), -1)
    else:
        mask = np.ones((rows, cols, 2), np.uint8)
        cv2.circle(mask, (ccol, crow), radius, (0, 0, 0), -1)

    fshift = dft_shift*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    img_back = np.uint8(img_back)

    return img_back

# 模板匹配有关函数
def Nms(boxes, scores, iou_threshold=0.5):
    """
    进行非极大值抑制(NMS)以去除重叠的边界框。
    参数:
        boxes:边界框列表,每个边界框由四个坐标值(x1, y1, x2, y2)组成。
        scores:每个边界框对应的匹配度分数列表。
        iou_threshold:交并比阈值,默认为0.5。交并比大于该阈值的边界框将被抑制。
    返回:
        保留的边界框索引列表。
    """
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    # 将匹配度降序排序,获取排序后的索引
    order = np.argsort(scores)[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # 计算当前最高匹配度的框与其他框的交并比
        xx1 = np.maximum(boxes[i][0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i][1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i][2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i][3], boxes[order[1:], 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / ( (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]) + (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1]) - inter + 1e-6)
        # 保留交并比小于阈值的框
        index = np.where(iou <= iou_threshold)[0]
        order = order[index + 1]
    return keep

def Template_Matching(img, templates, threshold=0.5, min_scale=0.5,num_scale=5):
    """
    在输入图像中查找与模板图像匹配的区域。
    参数:
        img: 输入图像，灰度图与彩色图均可。
        template: 模板图像，必须为灰度图。
        threshold:模板匹配阈值,默认为0.5。
        min_scale:在多尺度匹配中最小尺度,默认为0.5。
        num_scale:多尺度匹配中的尺度数量,默认为5。
    返回:
        valid_index:高于阈值的模板的索引。
        valid_location:模板在图中对应的坐标位置。
    """
    best_template = []
    best_box = []
    valid_index = []

    if img is None:
        raise ValueError("输入图像不能为空")
    if img.ndim not in [2, 3]:
        raise ValueError("输入图像必须为灰度图或彩色图")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 在min_scale和1.0之间生成一个包含 num_scale 个等间距数值的数组
    scales = np.linspace(min_scale, 1.0, num_scale)

    for template in templates:
        resize_template = template
        good_match = [] # 符合条件模板的匹配度
        good_box = [] # 符合条件的模板的边界框
        # 不同尺度进行匹配
        for scale in scales:
            resize_template = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            h, w = resize_template.shape
            # 进行模板匹配,返回模板匹配的相似度结果
            res = cv2.matchTemplate(img, resize_template, cv2.TM_CCOEFF_NORMED)
            locs = np.where(res >= threshold)
            for pt in zip(*locs[::-1]):
                x, y = pt[0], pt[1]
                good_box.append(np.asarray([x, y, x + w, y + h]))
                good_match.append(res[y, x])
        # 如果没有匹配到,就返回空列表，跳过NMS
        if not good_box:
            best_template.append([])
            best_box.append([])
            continue
        # 进行NMS
        keep = Nms(good_box, good_match, iou_threshold=0.5)
        keep_box = [good_box[i] for i in keep]
        keep_match = [good_match[i] for i in keep]
        # 将该模板的最高匹配度储存在数组中
        best_template.append(keep_match)
        best_box.append(keep_box)

    # 返回最高匹配度大于阈值的模板的index
    valid_index = [i for i, scores_list in enumerate(best_template) if scores_list]
    valid_box = [best_box[i] for i in valid_index]
    valid_match = [scores_list for i, scores_list in enumerate(best_template) if scores_list]

    return valid_index, valid_box, valid_match

def Create_Arr(*args):
    """
    创建用于模板匹配的数组。
    参数:
        *args:任意数量的同类型元素。
    返回:
        合成后的数组。
    """
    if not args:
        raise ValueError("输入不能为空")
    elements = []
    for element in args:
        element = cv2.cvtColor(element, cv2.COLOR_BGR2GRAY) if element.ndim == 3 else element
        element = cv2.normalize(element, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        elements.append(element)

    return elements

# 颜色提取相关函数
def Color_Extraction(img, color=RED):
    """
    提取图片中的特定颜色。
    参数:
        img:提取颜色的图像(BGR格式)。
        color:提取的颜色,默认为红色。
    返回:
        提取颜色后的BGR图像。
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if color == RED:
        mask = cv2.inRange(img_hsv, RED_LOWER1, RED_UPPER1) + cv2.inRange(img_hsv, RED_LOWER2, RED_UPPER2)
    elif color == GREEN:
        mask = cv2.inRange(img_hsv, GREEN_LOWER, GREEN_UPPER)
    elif color == YELLOW:
        mask = cv2.inRange(img_hsv, YELLOW_LOWER, YELLOW_UPPER)
    elif color == WHITE:
        mask = cv2.inRange(img_hsv, WHITE_LOWER, WHITE_UPPER)
    elif color == BLUE:
        mask = cv2.inRange(img_hsv, BLUE_LOWER, BLUE_UPPER)
    elif color == ORANGE:
        mask = cv2.inRange(img_hsv, ORANGE_LOWER, ORANGE_UPPER)
    elif color == PURPLE:
        mask = cv2.inRange(img_hsv, PURPLE_LOWER, PURPLE_UPPER)
    elif color == PINK:
        mask = cv2.inRange(img_hsv, PINK_LOWER, PINK_UPPER)
    elif color == BROWN:
        mask = cv2.inRange(img_hsv, BROWN_LOWER, BROWN_UPPER)
    elif color == GRAY:
        mask = cv2.inRange(img_hsv, GRAY_LOWER, GRAY_UPPER)
    elif color == BLACK:
        mask = cv2.inRange(img_hsv, BLACK_LOWER, BLACK_UPPER)
    else:
        raise ValueError("不支持的颜色类型")

    result = cv2.bitwise_and(img, img, mask=mask)

    return result

def Color_Extraction_Dynamic(img, hsv_lower, hsv_upper):
    """
    提取图片中的特定颜色(自定HSV阈值)
    参数:
        img:提取颜色的图像(BGR格式)。
        hsv_lower:HSV的下限阈值。
        hsv_upper:HSV的上限阈值。
    返回:
        提取颜色后的BGR图像。
    """
    img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img_hsv, hsv_lower, hsv_upper)
    result = cv2.bitwise_and(img, img, mask=mask)

    return result

# 中心点计算相关函数

def _cal_single_center(contour):
    """
    内部函数,计算轮廓的中心坐标。
    参数:
        contour:需要计算中心值的轮廓。
    返回:
        轮廓的中心x值、y值坐标。失败时返回-1, -1。
    """
    M = cv2.moments(contour)
    if M['m00'] > 0:
        return int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
    else:
        return -1, -1

def Get_Center_Point(contour, mode=CENTER_MAX):
    """
    提取轮廓中的中心值坐标。
    参数:
        contour:需要计算中心值的轮廓。CENTER_SINGLE模式下为单个轮廓,其他情况下为轮廓列表。
        mode:计算模式,有CENTER_SINGLE,CENTER_MAX和CENTER_ALL三个选项。
    返回:
        轮廓的中心x值、y值坐标。失败时返回-1, -1。
    """
    cx, cy = 0.0, 0.0
    if not contour:
        return -1, -1
    if mode == CENTER_SINGLE:
        return _cal_single_center(contour)
        
    elif mode == CENTER_MAX:
        max_contour = max(contour, key=cv2.contourArea)
        return _cal_single_center(max_contour)
    
    elif mode == CENTER_ALL:
        total_area = 0
        for cnt in contour:
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                total_area += M["m00"]
                cx += M['m10']
                cy += M['m01']
            else:
                continue
        if total_area != 0:
            cx = int(cx / total_area)
            cy = int(cy / total_area)
            return cx, cy
        else:
            return -1, -1
    else:
        raise ValueError("mode参数不存在")

# 特定多边形查找相关函数

def Find_Poly(contours, shape=4, min_area=None, max_area=None, factor=0.1):
    """
    对轮廓进行多边形逼近,筛选出特定形状的多边形。
    参数:
        contours:需要进行多边形逼近的轮廓列表。
        shape:需要找出的多边形边数,默认为4。
        min_area:轮廓最小面积,默认为None(无限制)。
        max_area:轮廓最大面积,默认为None(无限制)。
        factor:轮廓逼近时的参数,默认为0.1。
    返回:
        有符合条件的轮廓时返回符合条件轮廓顶点坐标的列表,没有符合条件的轮廓时返回空列表。
    """
    if shape < 3:
        raise ValueError("多边形边数不能小于3")
    if contours is None:
        return []
    if not contours:
        return []
    
    valid_contour_vertex = []

    for cnt in contours:
        epsilon = factor * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        contour_num = len(approx)
        contour_area = cv2.contourArea(cnt)
        if contour_num == shape:
            if (min_area is None or contour_area >= min_area) and (max_area is None or contour_area <= max_area):
                valid_contour_vertex.append(approx)
    return valid_contour_vertex

def Find_Circle(contours, min_area=None, max_area=None, factor=0.2):
    """
    筛选轮廓中接近圆形的轮廓。
    参数:
        contours:需要进行筛选的轮廓列表。
        min_area:轮廓最小面积,默认为None(无限制)。
        max_area:轮廓最大面积,默认为None(无限制)。
        factor:轮廓面积与外接圆面积的最大差异比例,默认为0.2。
    返回:
        有符合条件的轮廓时返回符合条件轮廓外接圆圆心坐标以及半径的列表,没有符合条件的轮廓时返回空列表。
    """
    if contours is None or not contours:
        return [], []
    
    valid_centers = []
    valid_radius = []

    for cnt in contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        circle_area = np.pi * radius * radius
        contour_area = cv2.contourArea(cnt)
        if contour_area != 0 and abs(circle_area - contour_area) / contour_area < factor:
            if (min_area is None or contour_area >= min_area) and (max_area is None or contour_area <= max_area):
                valid_centers.append((x, y))
                valid_radius.append(radius)
    return valid_centers, valid_radius

# 图像映射函数
def Perspective_Transform(img, box):
    width = int(max(
    np.linalg.norm(box[1] - box[0]),  # 上边宽度
    np.linalg.norm(box[2] - box[3])   # 下边宽度
                ))
    height = int(max(
    np.linalg.norm(box[3] - box[0]),  # 左边高度
    np.linalg.norm(box[2] - box[1])   # 右边高度
                ))
    pts_dst = np.float32([
        [0, 0],        # 左上
        [width - 1, 0],        # 右上
        [width - 1, height - 1], # 右下
        [0, height - 1]         # 左下
                ])
    M = cv2.getPerspectiveTransform(np.float32(box), pts_dst)
    warped = cv2.warpPerspective(img, M, (width, height))
    return warped

class YOLODetector:
    def __init__(self, model_path, num_classes, method="onnx", conf_thresh=0.5, iou_thresh=0.45, imgsz=(224,224), cores=None):
        self.method = method
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.imgsz = imgsz
        self.num_classes = num_classes
        self.cores = cores
        self.session = None
        self.rknn = None

    def __enter__(self):
        if self.method == "onnx":
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            if self.cores is not None:
                opts.intra_op_num_threads = self.cores
            try:
                self.session = ort.InferenceSession(self.model_path, sess_options=opts, providers=['CPUExecutionProvider'])
            except ort.ONNXRuntimeError as e:
                print(f"Error loading ONNX model: {e}")
                raise RuntimeError("Failed to load ONNX model.")
            except Exception as e:
                print(f"Unexpected error loading ONNX model: {e}")
                raise RuntimeError("Failed to load ONNX model due to an unexpected error.")
            inp = self.session.get_inputs()[0]
            self.input_name = inp.name
            self.output_name = self.session.get_outputs()[0].name
        elif self.method == "rknn":
            from rknnlite.api import RKNNLite
            self.rknn = RKNNLite()
            if self.rknn.load_rknn(self.model_path) != 0:
                raise RuntimeError("Failed to load RKNN model.")
            core_mask = getattr(RKNNLite, 'NPU_CORE_0_1_2', RKNNLite.NPU_CORE_0)
            if self.rknn.init_runtime(core_mask=core_mask) != 0:
                self.rknn.release()
                raise RuntimeError("Failed to initialize RKNN runtime.")
        else:
            raise ValueError("Unsupported method: {}".format(self.method))
        
        return self

    @staticmethod
    def letterbox_onnx(img, target_size):
        """保持长宽比缩放，并填充灰边(114)"""
        h, w = img.shape[:2]
        tw, th = target_size
        scale = min(tw / w, th / h)
        nw, nh = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (nw, nh))
        canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
        top = (th - nh) // 2
        left = (tw - nw) // 2
        canvas[top:top+nh, left:left+nw] = img_resized
        return canvas, scale, (left, top)
    
    @staticmethod
    def letterbox_rknn(img, new_shape=640, color=(114, 114, 114)):
        h, w = img.shape[:2]
        scale = min(new_shape / h, new_shape / w)
        nw, nh = int(w * scale), int(h * scale)
        canvas = np.full((new_shape, new_shape, 3), color, dtype=np.uint8)
        pad_w, pad_h = (new_shape - nw) // 2, (new_shape - nh) // 2
        canvas[pad_h:pad_h + nh, pad_w:pad_w + nw] = cv2.resize(img, (nw, nh))
        return canvas, scale, pad_w, pad_h

    @staticmethod
    def nms_onnx(boxes, scores, iou_thresh):
        """NMS算法"""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
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
            inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[np.where(iou <= iou_thresh)[0] + 1]
        return keep
    
    @staticmethod
    def nms_rknn(boxes, scores, iou_thresh):
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1: break
            xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
            xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
            inter = (xx2 - xx1).clip(min=0) * (yy2 - yy1).clip(min=0)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[1:][iou < iou_thresh]
        return keep

    @staticmethod
    def postprocess_onnx(output, conf_thresh, iou_thresh, num_classes, scale, pad):
        pred = output.squeeze(0).T  # [1, 12, 8400] -> [8400, 12]
        
        boxes = pred[:, :4]          # xywh (已经是 0-224 的绝对坐标)
        class_scores = pred[:, 4:4+num_classes]
        scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        mask = scores > conf_thresh
        boxes, scores, class_ids = boxes[mask], scores[mask], class_ids[mask]
        if len(boxes) == 0:
            return np.array([]), np.array([]), np.array([])

        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        keep = YOLODetector.nms_onnx(boxes_xyxy, scores, iou_thresh)
        boxes_xyxy, scores, class_ids = boxes_xyxy[keep], scores[keep], class_ids[keep]

        # 映射回原图坐标 (减去 pad，除以 scale)
        pad_left, pad_top = pad
        boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - pad_left) / scale
        boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - pad_top) / scale

        return boxes_xyxy.astype(int), scores, class_ids

    @staticmethod
    def postprocess_rknn(outputs, scale, pad_w, pad_h, conf_thresh, iou_thresh):
        pred = outputs[0][0].T 
        boxes_xywh = pred[:, :4]
        class_scores = pred[:, 4:]
        scores = class_scores.max(axis=1)
        class_ids = class_scores.argmax(axis=1)
        mask = scores > conf_thresh
        boxes_xywh, scores, class_ids = boxes_xywh[mask], scores[mask], class_ids[mask]
        if len(boxes_xywh) == 0: return [], [], []
        boxes_xyxy = np.zeros_like(boxes_xywh)
        boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
        boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
        boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - pad_w) / scale
        boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - pad_h) / scale
        keep = YOLODetector.nms_rknn(boxes_xyxy, scores,iou_thresh)
        return boxes_xyxy[keep].astype(np.int32), scores[keep], class_ids[keep]
    
    def detect(self, img):
        if self.method == "onnx":
            img_lb, scale, pad = YOLODetector.letterbox_onnx(img, self.imgsz)
            img_data = img_lb.astype(np.float32) / 255.0
            img_data = img_data.transpose(2, 0, 1)[np.newaxis, :]  # [1, 3, H, W]
            output = self.session.run([self.output_name], {self.input_name: img_data})[0]
            boxes, scores, class_ids = YOLODetector.postprocess_onnx(output, self.conf_thresh, self.iou_thresh, self.num_classes, scale, pad)
            return boxes, scores, class_ids
        elif self.method == "rknn":
            img, scale, pad_w, pad_h = YOLODetector.letterbox_rknn(img, self.imgsz[0])
            input_tensor = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), axis=0)
            outputs = self.rknn.inference(inputs=[input_tensor])
            boxes, scores, cls_ids = YOLODetector.postprocess_rknn(outputs, scale, pad_w, pad_h, self.conf_thresh, self.iou_thresh)
            return boxes, scores, cls_ids

    def draw_boxes(self, img, boxes, scores, class_ids):
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"ID: {class_id}, Score: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return img

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.method == "rknn" and self.rknn is not None:
            self.rknn.release()
            self.rknn = None
        elif self.method == "onnx" and self.session is not None:
            del self.session
            self.session = None
            import gc
            gc.collect()
        return False


def Sigmoid_Curve_Transform(image_gray, k=5.0, threshold=128):
    """
    带拐点控制的 S 形灰度映射（Sigmoid 型）
    让低于阈值的像素更黑，高于阈值的像素更白

    :param image_gray: 输入灰度图，uint8 类型，shape (H, W)
    :param k: 陡峭系数，推荐 2.0 ~ 10.0，越大过渡越锐利
    :param threshold: 拐点阈值（0~255），该灰度值处的输出正好为 128（中间灰）
                      例如 threshold=150 表示只有大于150的才会变白，小于150的变黑
    :return: 处理后的灰度图，uint8 类型
    """
    # 将阈值归一化到 [0,1]
    T = np.clip(threshold / 255.0, 0.01, 0.99)  # 避免极端值导致数值不稳定

    # 归一化像素值到 [0,1]
    img_float = image_gray.astype(np.float32) / 255.0

    # 计算 sigmoid 值
    S = 1.0 / (1.0 + np.exp(-k * (img_float - T)))
    # 端点归一化，保证输入 0 → 输出 0，输入 1 → 输出 1
    S0 = 1.0 / (1.0 + np.exp(k * T))          # 对应 x=0
    S1 = 1.0 / (1.0 + np.exp(-k * (1.0 - T))) # 对应 x=1
    transformed = (S - S0) / (S1 - S0)

    # 截断并转回 uint8
    transformed = np.clip(transformed, 0.0, 1.0)
    return (transformed * 255).astype(np.uint8)

def Sigmoid_Curve_Transform_LUT(image_gray, k=5.0, threshold=128):

    T = np.clip(threshold / 255.0, 0.01, 0.99)
    x = np.arange(256, dtype=np.float32) / 255.0
    
    S = 1.0 / (1.0 + np.exp(-k * (x - T)))
    S0 = 1.0 / (1.0 + np.exp(k * T))
    S1 = 1.0 / (1.0 + np.exp(-k * (1.0 - T)))
    
    lut = np.clip((S - S0) / (S1 - S0), 0.0, 1.0)
    lut = (lut * 255).astype(np.uint8)  # 长度为256的数组

    return cv2.LUT(image_gray, lut)
