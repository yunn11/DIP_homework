import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift, ifft2


image = cv2.imread('nchu_1112_4198_dip_hw_04_figs\Fig0526(a)(original_DIP).tif', cv2.IMREAD_GRAYSCALE)
image = image.astype(np.float64) 

def blur_filter(image):
    M, N = image.shape
    T = 1
    a = 0.1
    b = 0.1
    P = np.zeros((M,N), dtype=complex)
    Q = np.zeros((M,N), dtype=complex)
    for u in range(M):
        for v in range(N):
            P[u, v] = (np.pi * (a * u+ b * v))
            if P[u ,v] == 0:
                Q[u, v] = T
            else:
                Q[u, v] = T * np.sin(P[u, v]) * np.exp((-1j) * P[u, v]) /P[u,v]
    return Q

F = fftshift(fft2(image))
H = blur_filter(F)
G = H * F
image_ifft = (ifft2(fftshift(G)))
image_ifft = (np.abs(image_ifft)) 
print(image_ifft)
plt.imshow(image_ifft, cmap='gray'), plt.title('blurred image')
plt.show()  