# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 17:31:26 2020

@author: willr
"""
import numpy as np
import cv2 as cv2

import scipy.ndimage.interpolation as ndii
from scipy import interpolate

import matplotlib.pyplot as plt

import matplotlib.style
import matplotlib as mpl
import math

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rcParams['figure.figsize'] = [8.0, 6.0]
mpl.rcParams['figure.dpi'] = 400
mpl.rcParams['savefig.dpi'] = 450
mpl.rcParams['hatch.linewidth'] = 1.0
mpl.rcParams['font.size'] = 16
mpl.rcParams['legend.fontsize'] = 'large'
mpl.rcParams['figure.titlesize'] = 'medium'

from image_helper import *
import PhaseCorrelation 


plotting = True

#Load image and flatten
reference = cv2.imread('rome_image.png')
#reference = cv2.imread('Rome_Image.jpg')
#reference = cv2.imread('Rome_Image_Raster.tif')
reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

target = cv2.imread('cropped_intensity.jpg')
#target = cv2.imread('Rome_Height.jpg')
#target = cv2.imread('cropped_rome.jpg')
#target = cv2.imread('LiDAR_DSM.tiff')

target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

#temp = target
#target = reference
#reference = temp

#target, reference = padding(reference, target, dtype=np.uint8)
#target = warp(target, [0, 0, 0, 2.5, 2.46])

implot(reference)
implot(target)

angle_range = [-5, 5]
nbatch = [100, 100, 100, 100]
scales=[0.5, 0.5]
params = np.asarray([math.radians(angle_range[0]), math.radians(angle_range[1]), 20, 0.4, 2, 51, 0.4, 2, 51], dtype=np.float64)
t = multilevelRoutine(reference, target, params, nBatch=nbatch, scales=scales, system="GPU")

implot(reference)
implot(t)

t, reference = padding(reference, t, dtype=np.uint8)
added_image = cv2.addWeighted(reference, 0.5, t, 0.5, 0)

out = create_checker(reference)
A = reference*out + (1-out) *t
implot(A)