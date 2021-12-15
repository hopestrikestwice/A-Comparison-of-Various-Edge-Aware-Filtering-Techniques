import cv2
import numpy as np
from skimage import io
from matplotlib import pyplot as plt
from numpy.linalg.linalg import qr

def guidedFilter(p, I, r, eps):
    #1.
    mean_I = cv2.blur(I, (2*r + 1, 2*r + 1), borderType = cv2.BORDER_REPLICATE)
    mean_p = cv2.blur(p, (2*r + 1, 2*r + 1), borderType = cv2.BORDER_REPLICATE)
    corr_I = cv2.blur(I * I, (2*r + 1, 2*r + 1), borderType = cv2.BORDER_REPLICATE)
    corr_Ip = cv2.blur(I * p, (2*r + 1, 2*r + 1), borderType = cv2.BORDER_REPLICATE)

    #2.
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I  * mean_p

    #3.
    a = cov_Ip / (var_I + eps)
    print("a is", a)
    b = mean_p - a * mean_I
    print("b is", b)
    plt.imshow(b)
    plt.show()

    #4.
    mean_a = cv2.blur(a, (2*r + 1, 2*r + 1), borderType = cv2.BORDER_REPLICATE)
    mean_b = cv2.blur(b, (2*r + 1, 2*r + 1), borderType = cv2.BORDER_REPLICATE)

    #5.
    q = mean_a * I + mean_b

    return q

def guidedFilter_color(p, I, r, eps):
    """
    p : is single channel
    I : is color RGB image (3 channel)
    """
    hei = p.shape[0]
    wid = p.shape[1]

    #1.
    mean_I_r = cv2.blur(I[:, :, 0], (2*r + 1, 2*r + 1), borderType = cv2.BORDER_REPLICATE)
    mean_I_g = cv2.blur(I[:, :, 1], (2*r + 1, 2*r + 1), borderType = cv2.BORDER_REPLICATE)
    mean_I_b = cv2.blur(I[:, :, 2], (2*r + 1, 2*r + 1), borderType = cv2.BORDER_REPLICATE)

    mean_p = cv2.blur(p, (2*r + 1, 2*r + 1), borderType = cv2.BORDER_REPLICATE)

    mean_Ip_r = cv2.blur(I[:, :, 0] * p, (2*r + 1, 2*r + 1), borderType = cv2.BORDER_REPLICATE)
    mean_Ip_g = cv2.blur(I[:, :, 1] * p, (2*r + 1, 2*r + 1), borderType = cv2.BORDER_REPLICATE)
    mean_Ip_b = cv2.blur(I[:, :, 2] * p, (2*r + 1, 2*r + 1), borderType = cv2.BORDER_REPLICATE)

    print("1")
    #2.
    cov_Ip_r = mean_Ip_r - mean_I_r * mean_p
    cov_Ip_g = mean_Ip_g - mean_I_g * mean_p
    cov_Ip_b = mean_Ip_b - mean_I_b * mean_p
    
    var_I_rr = cv2.blur(I[:, :, 0] * I[:, :, 0], (2*r + 1, 2*r + 1), borderType = cv2.BORDER_REPLICATE) - mean_I_r * mean_I_r
    var_I_rg = cv2.blur(I[:, :, 0] * I[:, :, 1], (2*r + 1, 2*r + 1), borderType = cv2.BORDER_REPLICATE) - mean_I_r * mean_I_g
    var_I_rb = cv2.blur(I[:, :, 0] * I[:, :, 2], (2*r + 1, 2*r + 1), borderType = cv2.BORDER_REPLICATE) - mean_I_r * mean_I_b
    var_I_gg = cv2.blur(I[:, :, 1] * I[:, :, 1], (2*r + 1, 2*r + 1), borderType = cv2.BORDER_REPLICATE) - mean_I_g * mean_I_g
    var_I_gb = cv2.blur(I[:, :, 1] * I[:, :, 2], (2*r + 1, 2*r + 1), borderType = cv2.BORDER_REPLICATE) - mean_I_g * mean_I_b
    var_I_bb = cv2.blur(I[:, :, 2] * I[:, :, 2], (2*r + 1, 2*r + 1), borderType = cv2.BORDER_REPLICATE) - mean_I_b * mean_I_b

    print("2")
    #3.
    a = np.zeros((hei, wid, 3))
    for y in range(hei):
        print(y)
        for x in range(wid):
            Sigma = np.array([[var_I_rr[y, x], var_I_rg[y, x], var_I_rb[y, x]],
                              [var_I_rg[y, x], var_I_gg[y, x], var_I_gb[y, x]],
                              [var_I_rb[y, x], var_I_gb[y, x], var_I_bb[y, x]]])

            cov_Ip = np.array([[cov_Ip_r[y, x], cov_Ip_g[y, x], cov_Ip_b[y, x]]])

            a[y, x, :] = cov_Ip @ np.linalg.inv(Sigma + eps * np.eye(3))

    b = mean_p - a[:, :, 0] * mean_I_r - a[:, :, 1] * mean_I_g - a[:, :, 2] * mean_I_b

    #4.
    mean_a_r = cv2.blur(a[:, :, 0], (2*r + 1, 2*r + 1), borderType = cv2.BORDER_REPLICATE)
    mean_a_g = cv2.blur(a[:, :, 1], (2*r + 1, 2*r + 1), borderType = cv2.BORDER_REPLICATE)
    mean_a_b = cv2.blur(a[:, :, 2], (2*r + 1, 2*r + 1), borderType = cv2.BORDER_REPLICATE)

    mean_b = cv2.blur(b, (2*r + 1, 2*r + 1), borderType = cv2.BORDER_REPLICATE)

    #5.
    q = mean_a_r * I[:, :, 0] + mean_a_g * I[:, :, 1] + mean_a_b * I[:, :, 2] + mean_b

    return q


p = cv2.cvtColor(cv2.imread("../Pictures/cranepile.JPG"), cv2.COLOR_BGR2RGB).astype(np.double) / 255
I = cv2.cvtColor(cv2.imread("../Pictures/cranepile.JPG"), cv2.COLOR_BGR2RGB).astype(np.double) / 255
r = 8
eps = 0.4 ** 2

q = np.zeros_like(p)
q[:, :, 0] = guidedFilter_color(p[:, :, 0], I, r, eps)
q[:, :, 1] = guidedFilter_color(p[:, :, 1], I, r, eps)
q[:, :, 2] = guidedFilter_color(p[:, :, 2], I, r, eps)

io.imsave('../Results/Guided/ColorGuideColorBase/cranepile r=8 eps = 0.16.png', q)

plt.imshow(q)
plt.show()

plt.imshow(q - p)
plt.show()