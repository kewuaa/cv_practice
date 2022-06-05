# -*- coding: utf-8 -*-
#******************************************************************#
#
#                     Filename: main.py
#
#                       Author: kewuaa
#                      Created: 2022-05-23 12:32:47
#                last modified: 2022-06-05 19:30:36
#******************************************************************#
import pyximport
pyximport.install(language_level=3)
import image
import cv2
import numpy as np
img = cv2.imread('/mnt/e/PY/cv/project/apex_spotting/test.jpg')[..., 0]
new_img = image.gaussian_filter(img.astype(np.float64), 5, 5, 1.4)
cv2.imshow('img', new_img.astype(np.uint8))
cv2.waitKey(0)

