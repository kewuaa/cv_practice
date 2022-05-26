# -*- coding: utf-8 -*-
#******************************************************************#
#
#                     Filename: main.py
#
#                       Author: kewuaa
#                      Created: 2022-05-23 12:32:47
#                last modified: 2022-05-26 18:45:44
#******************************************************************#
import cv2
import numpy as np
import pyximport
pyximport.install(language_level=3)

import image


img = cv2.imread('../../digital_image_processing_course_design/data/camera.bmp')
# img = np.arange(36).reshape(6, 6)
new_img = image.median_filter(img.astype(np.float64), 3)
print(img, new_img, sep='\n')
cv2.imshow('img', new_img.astype(np.uint8))
cv2.waitKey(0)
# arr = np.arange(36, dtype=np.float64).reshape(6, -1)
# fft_arr_np = np.fft.fft2(arr)
# fft_arr = image.fft2(arr)
# print(fft_arr, fft_arr_np, sep='\n')
# print(image.ifft2(fft_arr))

