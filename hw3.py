import numpy as np
from scipy.fft import fft2, fftshift, ifft2
import matplotlib.pyplot as plt
import cv2

img1 = plt.imread('nchu_1112_4198_dip_hw_03_figs\Fig0441(a)(characters_test_pattern).tif')
ft = fft2(img1)
ft_shift = fftshift(ft)
magnitude_spectrum = np.abs(ft_shift)
magnitude_spectrum = np.log(1 + magnitude_spectrum)
M,N = img1.shape
sum = np.sum(np.abs(ft_shift))
avg = sum/(M*N)
print(avg) 
plt.imshow(magnitude_spectrum, cmap='gray'), plt.title('forier spectrum')
plt.show()

img2 = plt.imread('nchu_1112_4198_dip_hw_03_figs\Fig0235(c)(kidney_original).tif')
smoothed_img = cv2.GaussianBlur(img2, (5, 5), 2)
def sobel_gradient(img):
    mask_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    mask_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    Gx = cv2.filter2D(img, cv2.CV_64F, mask_x)
    Gy = cv2.filter2D(img, cv2.CV_64F, mask_y)
    G = abs(Gx)+abs(Gy)

    plt.hist(G.flatten(),256,[0,256], color = '#ff7389')
    plt.xlim([0,256])
    plt.show()
    return G

sobel = sobel_gradient(smoothed_img)
T = input('Enter the threshold:')
M,N = img2.shape
for i in range(M):
    for j in range(N):
        if sobel[i,j] < int(T): sobel[i,j] = 0
        else: sobel[i,j] = 255
plt.imshow(sobel, cmap='gray'), plt.title('edge detection')
plt.show()

img3 = plt.imread('nchu_1112_4198_dip_hw_03_figs/Fig0457(a)(thumb_print).tif')
M, N = img3.shape
P, Q = M*2, N*2
D0 = 50
highpass = np.zeros((P,Q))
for i in range(P):
    for j in range(Q):
        highpass[i,j] = (1 - np.exp(-((i-P/2)**2 + (j-Q/2)**2)/(2*D0**2)))
        
Forier = np.zeros((P,Q))
Forier[:M,:N] = img3[:M,:N]

for i in range(P):
    for j in range(Q):
        Forier[i,j] = Forier[i,j] * (-1)**(i+j)
Forier = (fft2(Forier))
Forier = Forier*highpass
Forier = ifft2(Forier).real
for i in range(P):
    for j in range(Q):
        Forier[i,j] = Forier[i,j] * (-1)**(i+j)

result = Forier[:M, :N]
plt.subplot(1,3,1),plt.imshow(highpass, cmap='gray'), plt.title('highpass filter')
plt.subplot(1,3,2),plt.imshow(result, cmap='gray'), plt.title('before threshold')


for i in range(M):
    for j in range(N):
        if result[i,j] < 10: result[i,j] = 0
        else: result[i,j] = 255
plt.subplot(1,3,3),plt.imshow(result, cmap='gray'), plt.title('Highpass')
plt.show()



