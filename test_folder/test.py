import cv2
import os
import sys

print("=== ���������Ϣ ===")
print(f"Python�汾: {sys.version}")
print(f"OpenCV�汾: {cv2.__version__}")
print(f"��ǰ����Ŀ¼: {os.getcwd()}")
print(f"Python·��: {sys.executable}")

# ����ļ��Ƿ����
image_path = "test_canny.jpg"  
print(f"�ļ��Ƿ����: {os.path.exists(image_path)}")
print(f"�ļ�����·��: {os.path.abspath(image_path)}")

# ���OpenCV��ȡ����
if os.path.exists(image_path):
    img = cv2.imread(image_path)
    print(f"ͼ���ȡ���: {img is not None}")
    if img is not None:
        print(f"ͼ��ߴ�: {img.shape}")
else:
    print("? �ļ������ڣ�����·��")