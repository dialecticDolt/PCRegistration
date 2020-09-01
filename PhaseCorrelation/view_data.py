# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 20:00:26 2020

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
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 450
mpl.rcParams['hatch.linewidth'] = 1.0
mpl.rcParams['font.size'] = 16
mpl.rcParams['legend.fontsize'] = 'large'
mpl.rcParams['figure.titlesize'] = 'medium'

def padding(reference, target, scaling = 1):
    reference_height, reference_width = reference.shape
    target_height, target_width = target.shape
    
    if reference_height%2 != 0:
        reference_height = reference_height - 1
    if reference_width%2 != 0:
        reference_width = reference_width - 1
        
    if target_height%2 != 0:
        target_height = target_height - 1
    if target_width%2 != 0:
        target_width = target_width - 1
        
    M = int(scaling* max([reference_height, reference_width, target_height, target_width]))
    print("Padding to size: ", M)
    
    if M%2 != 0:
        M -= 1
    
    reference = reference[:reference_height, :reference_width]
    target = target[:target_height, :target_width]
    
    k = 3
    const = (np.mean(reference[:k, :k]) + np.mean(reference[:k, -k:]) +np.mean(reference[-k:, :k]) + np.mean(reference[-k:, -k:]))//4
    
    dtype = np.complex64
    
    new_reference = np.zeros((M, M), dtype=dtype) + 0
    height_offset = (M-reference_height)//2
    width_offset = (M-reference_width)//2
    
    new_reference[height_offset:height_offset+reference_height, width_offset:width_offset+reference_width] = reference
    
    k = 3
    const = (np.mean(target[:k, :k]) + np.mean(target[:k, -k:]) +np.mean(target[-k:, :k]) + np.mean(target[-k:, -k:]))//4
    
    new_target = np.zeros((M, M), dtype=dtype) + 0
    height_offset = (M-target_height)//2
    width_offset = (M-target_width)//2
    
    new_target[height_offset:height_offset+target_height, width_offset:width_offset+target_width] = target
    
    return new_reference, new_target

def highpass(shape):
    x = np.outer(
        np.cos(np.linspace(-math.pi/2.0, math.pi/2.0, shape[0])),
        np.cos(np.linspace(-math.pi/2.0, math.pi/2.0, shape[1])))
    return (1.0 - x) * (2.0 - x)

def create_window(reference, plot=False):
    height, width = reference.shape
    hanning_window = cv2.createHanningWindow((width, height), cv2.CV_64F)
    #hanning_window = hanning_window[height//2, :]
    #hanning_window = np.outer(hanning_window, hanning_window)
    
    if plot:
        plt.contourf(hanning_window)
        plt.show()
        
    return hanning_window

def centered_difference(image, axis=0):
    difference_kernel = [1/12, -2/3, 0, 2/3, -1/12]
    n = len(difference_kernel)
    kernel = np.zeros((n, n), dtype=np.float32)
    difference_kernel = [1/12, -2/3, 0, 2/3, -1/12]
    if axis == 0:
        kernel[1, :] = difference_kernel
    else:
        kernel[:, 1] = difference_kernel

    dst = cv2.filter2D(image, -1, kernel)
    return dst
    
def sobel(image, axis=0):
    if axis == 0:
        dst = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    else:
        dst = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    return dst

def derivative(reference, target, method='Difference'):
    
    if method=='Sobel':
        GR_x = sobel(reference, axis=0)
        #GR_x = np.absolute(GR_x)
        
        GR_y = sobel(reference, axis=1)
        #GR_y = np.absolute(GR_y)
        
        GR = GR_x + 1j*GR_y
        
        GT_x = sobel(target, axis=0)
        #GT_x = np.absolute(GT_x)
        
        GT_y = sobel(target, axis=1)
        #GT_y = np.absolute(GT_y)
        
        GT = GT_x + 1j*GT_y
    elif method == 'Difference':
        GR_x = centered_difference(reference, axis=0)
        #GR_x = np.absolute(GR_x)
        
        GR_y = centered_difference(reference, axis=1)
        #GR_y = np.absolute(GR_y)
        
        GR = GR_x + 1j*GR_y
        
        GT_x = centered_difference(target, axis=0)
        #GT_x = np.absolute(GT_x)
        
        GT_y = centered_difference(target, axis=1)
        #GT_y = np.absolute(GT_y)
        
        GT = GT_x + 1j*GT_y
        
    """    
    topcorner = np.unravel_index(np.argmax(np.ravel(GR) > 0 ), shape = GR.shape)
    botcorner = np.unravel_index(np.argmax(np.ravel(np.flip(GR.T[:])) > 0 ), shape = GR.shape)
    
    hstart = topcorner[0]
    hstop = reference.shape[0] - botcorner[0]
    wstart = topcorner[1]
    wstop = reference.shape[1] - botcorner[0]
    
    #print(hstart, hstop)
    #print(wstart, wstop)
    
    k= 6
    #Top Boundary
    GR[hstart-k:hstart+k, wstart:wstop] = 0 + 1j*0
    
    #Left Boundary
    GR[hstart:hstop, wstart-k:wstart+k] = 0 + 1j*0
    
    #Right Boundary
    GR[hstart:hstop, wstop-k:wstop+k] = 0 + 1j*0
    
    #Bottom Boundary
    GR[hstop-k:hstop+k, wstart:wstop] = 0 + 1j*0
    
    topcorner = np.unravel_index(np.argmax(np.ravel(GT) > 0 ), shape = GT.shape)
    botcorner = np.unravel_index(np.argmax(np.ravel(np.flip(GT.T[:])) > 0 ), shape = GT.shape)
    
    hstart = topcorner[0]
    hstop = reference.shape[0] - botcorner[0]
    wstart = topcorner[1]
    wstop = reference.shape[1] - botcorner[0]

    #print(hstart, hstop)
    #print(wstart, wstop)

    #Top Boundary
    GT[hstart-k:hstart+k, wstart:wstop] = 0 + 1j*0
    
    #Left Boundary
    GT[hstart:hstop, wstart-k:wstart+k] = 0 + 1j*0
    
    #Right Boundary
    GT[hstart:hstop, wstop-k:wstop+k] = 0 + 1j*0
    
    #Bottom Boundary
    GT[hstop-k:hstop+k, wstart:wstop] = 0 + 1j*0
    
    """
    return GR, GT

def ft(reference, target, scaling=1, plot=False, windowing=True, label=""):
    
    if plot:
        fig = plt.figure()
        plt.imshow(reference.real)
        plt.title('FFT Setup: R (Before)'+str(label))
        plt.show()
    
    if windowing:
        window = create_window(reference)
        wref = reference*window
        
        window = create_window(target)
        wtar = target*window
    else:
        wref = reference
        wtar = target
    
    if plot:
        fig = plt.figure()
        plt.imshow(np.abs(wref))
        plt.title('FFT Setup: R (Window)'+str(label))
        plt.show()
    
    pref, ptar = padding(wref, wtar, scaling=scaling)
    
    if plot:
        fig = plt.figure()
        plt.imshow(np.abs(pref))
        plt.title('FFT Setup: R'+str(label))
        plt.show()
        
        fig = plt.figure()
        plt.imshow(np.abs(ptar))
        plt.title('FFT Setup: T'+str(label))
        plt.show()
    
    R_f = np.fft.fft2(pref)
    T_f = np.fft.fft2(ptar)
    
    if plot:
        fig = plt.figure()
        plt.imshow(np.abs(R_f))
        plt.title('FFT Setup: R_f'+str(label))
        plt.show()
        
        fig = plt.figure()
        plt.imshow(np.abs(T_f))
        plt.title('FFT Setup: T_f'+str(label))
        plt.show()
    
    return R_f, T_f


def phase_correlationCC(reference, target, plot=False, scaling=1, label="Phase Correlation (NGF)", windowing=False):
    
    eta = 0.001
    rg1 = np.real(reference)
    rg2 = np.imag(reference)
    
    norm = np.sqrt(rg1**2 + rg2**2 + eta)
    rg1 = rg1 / norm
    rg2 = rg2 / norm
    
    tg1 = np.real(target)
    tg2 = np.imag(target)
    
    norm = np.sqrt(tg1**2 + tg2**2 + eta)
    tg1 = tg1 / norm
    tg2 = tg2 / norm
    
    #Real Part FFT (squared)
    r1, t1 = ft(rg1*rg1, tg1*tg1, scaling = scaling, plot = plot, windowing=windowing, label=" sqr real ")
    
    #Imaginary Part FFT (squared)
    r2, t2 = ft(rg2*rg2, tg2*tg2, scaling = scaling, plot = plot, windowing=windowing,label=" sqr imag ")
    
    #Cross Terms
    cr, ct = ft(rg1*rg2, tg1*tg2, scaling = scaling, plot=plot, windowing=windowing, label=" cross ")
    
    #Normal (CC)
    nr1, tr1 = ft(rg1, tg1, scaling=scaling, plot=plot, windowing=windowing, label=" CC 1")
    nr2, tr2 = ft(rg2, tg2, scaling=scaling, plot=plot, windowing=windowing, label=" CC 2")
    
    corr_g1 = r1 * t1.conjugate()  #/ (np.abs(r1) * np.abs(t1.conjugate()))
    corr_g2 = r2 * t2.conjugate()  #/ (np.abs(r2) * np.abs(t2.conjugate()))
    corr_mix = cr * ct.conjugate() #/ (np.abs(cr) * np.abs(ct.conjugate()))
    
    cc1 = nr1*tr1.conjugate()
    cc2 = nr2*tr2.conjugate()
    
    #corr_g1  = np.fft.ifft2(corr_g1)
    #corr_g2  = np.fft.ifft2(corr_g2)
    #corr_mix = np.fft.ifft2(corr_mix)
    corr = np.fft.ifft2(corr_g1+corr_g2 + 2*corr_mix)
    
    cc_both = np.fft.ifft2(cc1) + np.fft.ifft2(cc2)

    print(corr_g1[0, 0])
    print(corr_g2[0, 0])
    print(corr_mix[0, 0])
    print(cc_both[0, 0])
    
    #corr = np.abs(corr_g1 + corr_g2 + 2*corr_mix)     
    #corr = np.abs(cc_both)
    corr= np.abs(corr)
    
    Dy, Dx = np.unravel_index(corr.argmax(), corr.shape)
    height, width = np.shape(corr)
    peak = corr[Dy, Dx]
    
    print(Dy, Dx)
    print(peak)
    
    if plot:
        fig = plt.figure()
        plt.imshow(np.log(corr+0.02))
        plt.title(label+': Correlation CC')
        plt.show()
    
    if(Dy > height//2):
        Dy -= height
        
    if(Dx > width//2):
        Dx -= width        
    
    return [Dy, Dx], peak, corr
    

def phase_correlation(reference, target, plot=False, scaling=1, label="Phase Correlation", windowing=False):
    
    R_f, T_f = ft(reference, target, scaling=scaling, plot=plot, windowing=windowing)
    aR_f, aT_f = ft(np.abs(reference), np.abs(target), scaling=scaling, plot=plot, windowing=windowing)
    
    height, width = R_f.shape
    
    eps = 0#1e-10
    thres = 0
    
    corr_f = R_f * T_f.conjugate()
    corr_f /= ( np.absolute(R_f) * np.absolute(T_f.conjugate()) + eps )
    corr_f = corr_f * (np.absolute(R_f) > thres) * (np.absolute(T_f.conjugate()) > thres)
    corr = np.abs(np.fft.ifft2(corr_f))
    
    #corr_f = R_f * T_f.conjugate()
    corr_f = np.fft.ifft2(corr_f)
    
    norm = aR_f * aT_f.conjugate()
    norm = np.fft.ifft2(norm)
    
    #corr = np.abs(corr_f)#/np.abs(norm)
    
    print("corr", np.min(corr_f))
    print("norm", np.min(norm) )
    
    Dy, Dx = np.unravel_index(corr.argmax(), corr.shape)
    
    peak = corr[Dy, Dx]
    
    if plot:
        fig = plt.figure()
        plt.imshow(np.log(np.abs(R_f)))
        plt.title(label+': FFT R')
        plt.show()
        
        fig = plt.figure()
        plt.imshow(np.log(np.abs(R_f)))
        plt.title(label+': FFT R')
        plt.show()
        
        fig = plt.figure()
        plt.semilogy(np.abs(R_f[height//2, :]))
        plt.title(label+': FFT T')
        plt.show()
        
        fig = plt.figure()
        plt.imshow(np.log(corr+0.02))
        plt.title(label+': Correlation')
        plt.show()
    
    print(Dy, Dx)
    print(peak)
    
    if(Dy > height//2):
        Dy -= height
        
    if(Dx > width//2):
        Dx -= width        
    
    return [Dy, Dx], peak, corr

def loglog(image, ovrsamp=None, res=(None, None), log_base=(None, None)):
    height, width = image.shape
    
    res_h = res[0]
    res_w = res[1]
    
    log_base_h = log_base[0]
    log_base_w = log_base[1]
    
    if ovrsamp is None:
        ovrsamp = 1
    
    if res_h is None:
        res_h = height*ovrsamp
    
    if res_w is None:
        res_w = width*ovrsamp
        
    if log_base_h is None:
        mr = np.log10(height/2)/res_h
        log_base_h = 10**(mr)
        
    if log_base_w is None:
        mr = np.log10(width/2)/res_w
        log_base_w = 10**(mr)
        
    lh = np.empty(res_h)
    lh[:] = np.power(log_base_h, np.linspace(0, res_h, num=res_h, endpoint=False))
    
    lw = np.empty(res_w)
    lw[:] = np.power(log_base_w, np.linspace(0, res_w, num=res_w, endpoint=False))
    
    eh = height//2
    oh = np.linspace(0, eh, num=eh, endpoint=False)
    
    ew = width//2
    ow = np.linspace(0, ew, num=ew, endpoint=False)
    print(ow)
    
    #image = np.flip(image[:eh, :ew])
    image = image[:eh, :ew]
    
    of = interpolate.interp2d(oh, ow, image, kind='linear')
    log_interp = of(lh, lw)
    
    #log_interp = ndii.map_coordinates(image, [lh, lw])
    print(image.shape)
    print(log_interp.shape)
    
    return log_interp, (log_base_h, log_base_w)

def poc2warp(center,param):
    cx,cy = center
    dx,dy,theta,scalex, scaley = param
    cs = math.cos(theta)
    sn = math.sin(theta)
    
    Rot = np.float32([[scalex* cs, scaley*sn, 0],[-scalex*sn, scaley*cs,0],[0,0,1]])
    center_Trans = np.float32([[1,0,cx],[0,1,cy],[0,0,1]])
    center_iTrans = np.float32([[1,0,-cx],[0,1,-cy],[0,0,1]])
    cRot = np.dot(np.dot(center_Trans,Rot),center_iTrans)
    Trans = np.float32([[1,0,dx],[0,1,dy],[0,0,1]])
    Affine = np.dot(cRot,Trans)
    
    return Affine

def warp(Img,param):
    center = np.array(Img.shape)/2
    rows,cols = Img.shape
    Affine = poc2warp(center,param) 
    #k = 3
    #const = (np.mean(Img[:k, :k]) + np.mean(Img[:k, -k:]) +np.mean(Img[-k:, :k]) + np.mean(Img[-k:, -k:]))//4
    const = 0
    outImg = cv2.warpPerspective(Img, Affine, (cols,rows), cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=const)
    return outImg

#Load Reference
R = np.fromfile('RTest.bin', dtype=np.uint8);
R = np.asarray(R, dtype=np.float64);
R = R.reshape(512, 512)
gR, gR2 = derivative(R, R);
plt.imshow(np.abs(gR))

#Load Reference Gradient
#fig = plt.figure()
RG= np.fromfile('RGTest.bin', dtype=np.complex64);
RG = RG.reshape(512, 512)

#plt.imshow(np.abs(R));
#pyFR = np.fft.ifft2(R);
#plt.imshow(np.abs(pyFR));

#fig = plt.figure()
FR = np.fromfile('RFTTest.bin', dtype=np.complex64);
FR = FR.reshape(512, 512)
#plt.imshow(np.abs(FR));
#print(FR);

#Load Moving Gradient
#fig = plt.figure()
TG = np.fromfile('Gtest.bin', dtype=np.complex64);
TG = TG.reshape(512, 512);

fig = plt.figure()
C= np.fromfile('PCtest.bin', dtype=np.complex64);
C = C.reshape(512, 512)
plt.imshow(np.abs(C));
Dy, Dx = np.unravel_index(C.argmax(), C.shape)

loc, peak, Corr = phase_correlation(RG, TG, plot=True)
print(loc)

eps = 0;
thres = 0;
R_f, T_f = ft(RG, TG, scaling=1, plot=True, windowing=False)
corr_f = R_f * T_f.conjugate()
corr_f /= ( np.absolute(R_f) * np.absolute(T_f.conjugate()) + eps )
corr_f = corr_f * (np.absolute(R_f) > thres) * (np.absolute(T_f.conjugate()) > thres)
corr = np.abs(np.fft.ifft2(corr_f))

#corr_f = R_f * T_f.conjugate()
corr_f = np.fft.ifft2(corr_f)
D = np.abs(corr_f)

    