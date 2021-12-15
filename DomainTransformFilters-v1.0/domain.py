import cv2
import numpy as np
from numpy.core.numeric import zeros_like
import scipy.ndimage
from skimage import io
from matplotlib import pyplot as plt

def NC(img, sigma_s, sigma_r, num_iterations = None, joint_image = None):
    I = img.astype(np.double)
    
    if num_iterations is None:
        num_iterations = 3
    if joint_image is None:
        J = I
    
    h = J.shape[0]
    w = J.shape[1]
    num_joint_channels = J.shape[2]

    #Compute domain transform
    dIcdx = np.diff(J, 1, 1)
    dIcdy = np.diff(J, 1, 0)
    
    dIdx = np.zeros((h, w))
    dIdy = np.zeros((h, w))

    for c in range(num_joint_channels):
        dIdx[:, 1:] = dIdx[:, 1:] + np.absolute(dIcdx[:, :, c])
        dIdy[1:, :] = dIdy[1:, :] + np.absolute(dIcdy[:, :, c])

    dHdx = (1 + sigma_s / sigma_r * dIdx)
    dVdy = (1 + sigma_s / sigma_r * dIdy)

    ct_H = np.cumsum(dHdx, 1)
    ct_V = np.cumsum(dVdy, 0)

    assert(len(ct_V.shape) == 2)
    ct_V = ct_V.T

    #Perform filtering
    N = num_iterations
    F = I

    sigma_H = sigma_s

    for i in range(num_iterations):
        print(i)
        sigma_H_i = sigma_H * np.sqrt(3) * pow(2, N - (i + 1)) / np.sqrt(4 ^ N - 1)
        box_radius = np.sqrt(3) * sigma_H_i

        F = transformedDomainBoxFilter_Horizontal(F, ct_H, box_radius)
        F = image_transpose(F)

        F = transformedDomainBoxFilter_Horizontal(F, ct_V, box_radius)
        F = image_transpose(F)

    F = F.astype(img.dtype)

    return F

def transformedDomainBoxFilter_Horizontal(I, xform_domain_position, box_radius):
    h = I.shape[0]
    w = I.shape[1]
    num_channels = I.shape[2]

    l_pos = xform_domain_position - box_radius
    u_pos = xform_domain_position + box_radius

    l_idx = np.zeros_like(xform_domain_position, dtype = int)
    u_idx = np.zeros_like(xform_domain_position, dtype = int)

    for row in range(h):
        print(row)
        xform_domain_pos_row = np.append(xform_domain_position[row, :], np.inf)

        l_pos_row = l_pos[row, :]
        u_pos_row = u_pos[row, :]

        local_l_idx = np.zeros((1, w), dtype = int)
        local_u_idx = np.zeros((1, w), dtype = int)

        local_l_idx[:, 0] = np.argwhere(np.greater(xform_domain_pos_row, l_pos_row[0]).astype(int).T.flatten())[0]
        local_u_idx[:, 0] = np.argwhere(np.greater(xform_domain_pos_row, u_pos_row[0]).astype(int).T.flatten())[0]

        for col in range(1, w):
            local_l_idx[:, col] = local_l_idx[:, col - 1] + np.argwhere((xform_domain_pos_row[local_l_idx[:, col - 1][0]:] > l_pos_row[col]).T.flatten())[0]
            local_u_idx[:, col] = local_u_idx[:, col - 1] + np.argwhere((xform_domain_pos_row[local_u_idx[:, col - 1][0]:] > u_pos_row[col]).T.flatten())[0]

        l_idx[row, :] = local_l_idx
        u_idx[row, :] = local_u_idx
    
    SAT = np.zeros((h, w + 1, num_channels))
    SAT[:, 1:, :] = np.cumsum(I, 1)
    row_indices = np.tile(np.arange(h)[:, np.newaxis], (1, w))
    F = np.zeros_like(I)

    for c in range(num_channels):
        # a = np.ravel_multi_index((row_indices, l_idx, np.tile(c, (h, w))), dims = SAT.shape, order = 'F')
        # b = np.ravel_multi_index((row_indices, u_idx, np.tile(c, (h, w))), dims = SAT.shape, order = 'F')
        # F[:, :, c] = (SAT[b] - SAT[a]) / (u_idx - l_idx)
        F[:, :, c] = (SAT[row_indices, u_idx, c] - SAT[row_indices, l_idx, c]) / (u_idx - l_idx)

    return F

def image_transpose(I):
    h = I.shape[0]
    w = I.shape[1]
    num_channels = I.shape[2]

    T = np.zeros((w, h, num_channels)).astype(I.dtype)

    for c in range(num_channels):
        T[:, :, c] = I[:, :, c].T
    
    return T


image = cv2.cvtColor(cv2.imread("../Pictures/towel.JPG"), cv2.COLOR_BGR2RGB).astype(np.double) / 255

sigma_s = 2
sigma_r = 0.2

F_nc = NC(image, sigma_s, sigma_r)

plt.imshow(F_nc)
plt.show()

plt.imshow((F_nc * 255).astype(int))

io.imsave('../Results/Domain/Normalized/cards sigs=2 sigr=0.2.png', F_nc)
