import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
import process_lib.image_lib as lb
import pydirectinput as pdi
import time

# img = cv2.imread("test_canny.png")
# red = lb.Color_Extraction(img, lb.WHITE)
# print(red.shape)
# print(np.sum(red))
# cv2.imshow("img", img)
# cv2.imshow("red", red)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
start_time = time.time()
while time.time() - start_time < 15:
    pdi.click(1280, 800, button='left')
    time.sleep(1)