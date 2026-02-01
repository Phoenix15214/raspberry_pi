import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
import lib.lib as lb


img = cv2.imread("test_canny.png")
high_pass_filter = lb.FFT_Filtering(img, radius=60, mode=lb.FFT_LOWPASS)
cv2.imshow("high_pass_filtering", high_pass_filter)
cv2.waitKey(0)
cv2.destroyAllWindows()
