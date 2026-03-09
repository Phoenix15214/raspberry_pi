# process_lib内容及使用方法

本文件中包含库的主要内容以及使用方法的说明以及注意事项。



## 一、使用方法

首先在库的上一级目录**（含有setup.py的目录）**中打开终端，激活运行时的环境，然后在终端运行:

`pip install -e .`

即可将库安装到运行环境。

导入时建议**使用别名**导入，如：

`import process_lib.image_lib as ib`

即可使用库中的函数，用法类似`numpy`。

## 二、process_lib内容

库中包含 **image_lib.py** 以及 **control_lib.py** ，前者用于处理图像，后者用于

控制、通信等功能。

## 三、process_lib使用方法

### image_lib.py

1. `FFT_Filtering(img, radius=30, mode=FFT_HIGHPASS)`

   对输入图像进行高通滤波处理，增强图像的边缘和细节。默认高通滤波。

   **参数:**

   - `img`: 输入图像，灰度图与彩色图均可。

   - `radius`: 高通滤波器的半径，默认为30。

   **返回:**处理后的图像。

2. `Template_Matching(img, templates, threshold=0.5, min_scale=0.7,num_scale=5, method=cv2.TM_CCOEFF_NORMED)`

   在输入图像中查找与模板图像匹配的区域。

   **参数:**

   - `img`: 输入图像，灰度图与彩色图均可。

   - ` template`: 模板图像，必须为灰度图。

   - ` threshold`:模板匹配阈值,默认为0.5。

   - `min_scale`:在多尺度匹配中最小尺度,默认为0.7。

   - `num_scale`:多尺度匹配中的尺度数量,默认为5。

   - `method`: 模板匹配方法,默认为cv2.TM_CCOEFF_NORMED。

   **返回:**最佳匹配结果在模板中的索引。

3. `Create_Arr(*args)`

   创建任意类型的数组(用于模板匹配中模板列表的创建)。

   **参数:**

   - `*args`:任意数量的同类型元素。

   **返回:**合成后的数组。

4. `Color_Extraction(img, color=RED)`

   提取图片中的特定颜色。

   **参数:**

   - `img`:提取颜色的图像(BGR格式)。

   - `color`:提取的颜色,默认为红色。

   **返回:**提取颜色后的BGR图像。

5. `Color_Extraction_Dynamic(img, hsv_lower, hsv_upper)`

   提取图片中的特定颜色(自定HSV阈值)

   **参数:**

   - `img`:提取颜色的图像(BGR格式)。

   - `hsv_lower`:HSV的下限阈值。

   - `hsv_upper`:HSV的上限阈值。

   **返回:**提取颜色后的BGR图像。

6. `Get_Center_Point(contour, mode=CENTER_MAX)`

   提取轮廓中的中心值坐标。

   **参数:**

   - `contour`:需要计算中心值的轮廓。CENTER_SINGLE模式下为单个轮廓,其他情况下为轮廓列表。
   - `mode`:计算模式,有CENTER_SINGLE,CENTER_MAX和CENTER_ALL三个选项。

   **返回:**轮廓的中心x值、y值坐标。失败时返回-1, -1。

7. `Find_Poly(contours, shape=4, min_area=None, max_area=None, factor=0.1)`

   对轮廓进行多边形逼近,筛选出特定形状的多边形。

   **参数:**

   - `contours`:需要进行多边形逼近的轮廓列表。

   - `shape`:需要找出的多边形边数,默认为4。

   - `min_area`:轮廓最小面积,默认为None(无限制)。

   - `max_area`:轮廓最大面积,默认为None(无限制)。

   - `factor`:轮廓逼近时的参数,默认为0.1。

   **返回:**有符合条件的轮廓时返回符合条件轮廓的列表,没有符合条件的轮廓时返回空列表。

8. `Find_Circle(contours, min_area=None, max_area=None, factor=0.2)`

   筛选轮廓中接近圆形的轮廓。

   **参数:**

   - `contours`:需要进行筛选的轮廓列表。

   - `min_area`:轮廓最小面积,默认为None(无限制)。

   - `max_area`:轮廓最大面积,默认为None(无限制)。

   - `factor`:轮廓面积与外接圆面积的最大差异比例,默认为0.2。

   **返回:**有符合条件的轮廓时返回符合条件轮廓的列表,没有符合条件的轮廓时返回空列表。