import cv2
import numpy as np
import scipy.ndimage
from skimage import io
from matplotlib import pyplot as plt

def lapfilter(I, sigma_r, alpha, beta, colorRemapping, domain):
    if I.shape[2] == 1:
        I = np.stack((I, I, I), axis = -1)

    noise_level = 0.01
    
    def fd(d):
        out = np.power(d, alpha)
        if alpha < 1:
            tau = smooth_step(noise_level, 2 * noise_level, d * sigma_r)
            out = tau * out + (1 - tau) * d   
        return out

    def fe(a):
        out = beta * a
        return out
    
    def r(i, g0):
        return r_color(i, g0, sigma_r, fd, fe)

    def to_domain(I):
        return I
    def from_domain(R):
        return R
    
    if alpha == 1 and beta == 1:
        R = I
    else:
        I = to_domain(I)
        R = lapfilter_core(I, r)
        R = from_domain(R)
    
    R = np.maximum(0, R)
    if beta <= 1:
        R = np.minimum(1, R)

    return R

def lapfilter_core(I, r):
    G = gaussian_pyramid(I)
    L = laplacian_pyramid(np.zeros_like(I))

    #O(N * log(N)) implementation
    for lev0 in range(len(L) - 1):
        print(lev0)
        hw = 3 * pow(2, lev0 + 1) - 2
        for y0 in range(G[lev0].shape[0]):
            for x0 in range(G[lev0].shape[1]):
                yf = y0 * pow(2, lev0)
                xf = x0 * pow(2, lev0)

                yrng = np.array([max(0, yf - hw), min(I.shape[0] - 1, yf + hw)])
                xrng = np.array([max(0, xf - hw), min(I.shape[1] - 1, xf + hw)])
                Isub = I[yrng[0]:yrng[1] + 1, xrng[0]:xrng[1] + 1, :]

                g0 = G[lev0][y0,x0,:]
                Iremap = r(Isub, g0)
                Lremap = laplacian_pyramid(Iremap, lev0 + 2, np.append(yrng, xrng))

                yfc = yf - yrng[0]
                xfc = xf - xrng[0]
                yfclev0 = np.floor(yfc / pow(2, lev0)).astype(int)
                xfclev0 = np.floor(xfc / pow(2, lev0)).astype(int)

                L[lev0][y0, x0, :] = Lremap[lev0][yfclev0, xfclev0, :]

    #O(N^2) implementation
    # for lev0 in range(len(L) - 1):
    #     print(lev0)
    #     for y0 in range(G[lev0].shape[0]):
    #         for x0 in range(G[lev0].shape[1]):
    #             print(x0)
    #             g0 = G[lev0][y0, x0, :]
    #             Iremap = r(I, g0)
    #             Lremap = laplacian_pyramid(Iremap, lev0 + 2)
    #             L[lev0][y0, x0, :] = Lremap[lev0][y0, x0, :]
    
    L[-1] = G[-1]
    R = reconstruct_laplacian_pyramid(L)
    
    return R

def gaussian_pyramid(I, nlev = None, subwindow = None):
    r = I.shape[0]
    c = I.shape[1]

    if subwindow is None:
        subwindow = np.array([1, r, 1, c])
    if nlev is None:
        nlev = numlevels([r, c])

    pyr = [None] * nlev
    pyr[0] = I

    filter = pyramid_filter()
    for level in range(1, nlev):
        I = downsample(I, filter, subwindow)[0]
        pyr[level] = I
    
    return pyr

def laplacian_pyramid(I, nlev = None, subwindow = None):
    r = I.shape[0]
    c = I.shape[1]
    if subwindow is None:
        subwindow = np.array([1, r, 1, c])
    if nlev is None:
        nlev = numlevels([r, c])

    pyr = [None] * nlev
    filter = pyramid_filter()
    J = I
    for level in range(nlev - 1):
        I, subwindow_child = downsample(J, filter, subwindow)
        pyr[level] = J - upsample(I, filter, subwindow)
        J = I
        subwindow = subwindow_child
    
    pyr[-1] = J

    return pyr

def reconstruct_laplacian_pyramid(pyr, subwindow = None):
    r = pyr[0].shape[0]
    c = pyr[0].shape[1]
    nlev = len(pyr)

    subwindow_all = np.zeros((nlev, 4))
    if subwindow is None:
        subwindow_all[0, :] = np.array([1, r, 1, c])
    else:
        subwindow_all[0, :] = subwindow

    for lev in range(1, nlev):
        subwindow_all[lev, :] = child_window(subwindow_all[lev - 1, :])
    
    R = pyr[nlev - 1]
    filter = pyramid_filter()
    for lev in range(nlev - 2, -1, -1):
        R = pyr[lev] + upsample(R, filter, subwindow_all[lev, :])

    return R

def numlevels(im_sz):
    """
    im_sz : (2,) vector defining [r, c] of image

    return : maximum levels in pyramid of given image size, up to 1x1
    """
    min_d = np.min(im_sz)
    nlev = 1
    while min_d > 1:
        nlev = nlev + 1
        min_d = (min_d + 1) // 2
    
    return nlev

def pyramid_filter():
    return np.array([[0.0025, 0.0125, 0.0200, 0.0125, 0.0025],
                     [0.0125, 0.0625, 0.1000, 0.0625, 0.0125],
                     [0.0200, 0.1000, 0.1600, 0.1000, 0.0200],
                     [0.0125, 0.0625, 0.1000, 0.0625, 0.0125],
                     [0.0025, 0.0125, 0.0200, 0.0125, 0.0025]])

def downsample(I, filter, subwindow = None):
    """
    I : image to downsample
    filter : 2D separable filter to use in downsampling
    """
    r = I.shape[0]
    c = I.shape[1]

    if subwindow is None:
        subwindow = np.array([1, r, 1, c])

    subwindow_child = child_window(subwindow)

    R = imfilterRGB(I, filter)
    Z = imfilterRGB(np.ones_like(I), filter)
    R = R / Z

    reven = int(subwindow[0] % 2 == 0)
    ceven = int(subwindow[2] % 2 == 0)
    R = R[reven:r:2, ceven:c:2, :]

    return R, subwindow_child

def upsample(I, filter, subwindow):
    r = int(subwindow[1] - subwindow[0] + 1)
    c = int(subwindow[3] - subwindow[2] + 1)
    k = I.shape[2]
    reven = int(subwindow[0] % 2 == 0)
    ceven = int(subwindow[2] % 2 == 0)

    R = np.zeros((r, c, k))
    R[reven:r:2, ceven:c:2, :] = I
    R = imfilterRGB(R, filter)

    Z = np.zeros((r, c, k))
    Z[reven:r:2, ceven:c:2, :] = 1
    Z = imfilterRGB(Z, filter)

    R = R / Z

    return R

def child_window(parent, N = None):
    """
    parent : (4,) numpy array of [r1, r2, c1, c2]

    return : corresponding child window of given parent window, N levels up the pyramid
    """
    if N is None:
        N = 1
    
    child = parent
    for K in range(N):
        child = (child + 1) / 2
        child[0] = np.ceil(child[0])
        child[2] = np.ceil(child[2])
        child[1] = np.floor(child[1])
        child[3] = np.floor(child[3])
    
    return child

def imfilterRGB(image, filter):
    """
    Function mimicking Matlab's imfilter for multi-channel images with default settings
    """
    res = np.zeros_like(image)
    for channel in range(image.shape[2]):
        res[:, :, channel] = scipy.ndimage.correlate(image[:, :, channel], filter, mode='constant')

    return res

def smooth_step(xmin, xmax, x):
    y = (x - xmin) / (xmax - xmin)
    y = max(0, min(1, y))
    y = np.power(y, 2) * np.power(y - 2, 2)
    return y

def r_color(i, g0, sigma_r, fd, fe):
    g0 = np.reshape(g0, (1, 1, 3))
    g0 = np.tile(g0, (i.shape[0], i.shape[1], 1))
    dnrm = np.sqrt(np.sum(np.power(i-g0, 2), 2))

    unit_pre = np.finfo(float).eps + dnrm
    unit = (i - g0) / np.stack((unit_pre, unit_pre, unit_pre), -1)

    rd_pre = sigma_r * fd(dnrm / sigma_r)
    rd = g0 + unit * np.stack((rd_pre, rd_pre, rd_pre), -1)
    re_pre = fe(dnrm - sigma_r) + sigma_r
    re = g0 + unit * np.stack((re_pre, re_pre, re_pre), -1)
    
    isedge_pre = dnrm > sigma_r
    isedge = np.stack((isedge_pre, isedge_pre, isedge_pre), -1)
    inew = np.logical_not(isedge) * rd + isedge * re

    return inew

image = cv2.cvtColor(cv2.imread("../Pictures/cranepile_downsized.png"), cv2.COLOR_BGR2RGB).astype(np.double) / 255

sigma_r = 0.4
alpha = 1.4
beta = 1
colorRemapping = 'rgb'
domain = 'lin'
R = lapfilter(image, sigma_r, alpha, beta, colorRemapping, domain)

io.imsave('../Results/Laplacian/cranepile A=1.4 sigma_r=0.4.png', R)

plt.imshow(R)
plt.show()