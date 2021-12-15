import cv2
import numpy as np
from skimage import io
from scipy import interpolate
import matplotlib.pyplot as plt

#must input images normalized to [0, 1]
def bilateralFilter(ambient, flash, sig_s, sig_r, lamb):
    minI = np.nanmin(flash) - lamb
    maxI = np.nanmax(flash) + lamb
    NB_SEGMENTS = np.ceil((maxI - minI) / sig_r)

    result = np.zeros_like(ambient)

    for channel in range(3):
        ambient_channel = ambient[:, :, channel]
        flash_channel = flash[:, :, channel]

        points_z = np.zeros(int(NB_SEGMENTS) + 1)
        J_all = np.zeros((ambient.shape[0], ambient.shape[1], int(NB_SEGMENTS) + 1))
        for j in range(int(NB_SEGMENTS) + 1):
            i_j = minI + j * (maxI - minI) / NB_SEGMENTS
            G_j = np.exp(-1 * np.square(flash_channel - i_j) / (2 * sig_r * sig_r)) #apply gauss function on image
            K_j = cv2.GaussianBlur(G_j, ksize = (0, 0), sigmaX = sig_s)
            H_j = G_j * ambient_channel
            H_dot_j = cv2.GaussianBlur(H_j, ksize = (0, 0), sigmaX = sig_s)
            J_j = H_dot_j / K_j
            J_all[:, :, j] = J_j

            points_z[j] = i_j

        q_j, q_i = np.meshgrid(np.arange(ambient.shape[1]), np.arange(ambient.shape[0]))
        xi = np.zeros_like(ambient)
        xi[:, :, 0] = q_i
        xi[:, :, 1] = q_j
        xi[:, :, 2] = flash_channel #I think this is flash here?
        points = [np.arange(J_all.shape[0]), np.arange(J_all.shape[1]), points_z]
        interpWeights = interpolate.interpn(points, J_all, xi)
        result[:, :, channel] = interpWeights

        print(channel)

    return result

#INPUT
ambient = io.imread('../Pictures/treepile1.JPG')
ambient = ambient / 255 #normalize
flash = io.imread('../Pictures/treepile1.JPG')
flash = flash / 255 #normalize

sig_s = 4 #test between [1, 64]
sig_r = 0.2 #test between [0.05, 0.25]
lamb = 0.001 #needs to be less than sig_r

#Basic Bilateral Filter
A_base = bilateralFilter(ambient, ambient, sig_s, sig_r, lamb)
io.imsave('../Results/Bilateral/treepile1 sigs=' + str(sig_s) + ' sigr=' + str(sig_r) + '.png', np.clip(A_base, 0, 1))
