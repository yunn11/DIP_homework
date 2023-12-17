import matplotlib.pyplot as plt
import numpy as np
import cv2

#8. Pseudo-Color Image Processing
img1 = cv2.imread('nchu_1112_4198_dip_hw_05_figs\Fig0110.TIF', cv2.IMREAD_GRAYSCALE)

M, N = img1.shape
R = np.zeros((M, N))
G = np.zeros((M, N))
B = np.zeros((M, N))
img1_processed = np.zeros((M,N,3))

for i in range(M):
    for j in range(N):
        if img1[i, j] < 255//5:
            img1_processed[i,j,2] = 1
            img1_processed[i,j,1] = 1
            img1_processed[i,j,0] = 0
        if img1[i, j] >= 255//5:
            img1_processed[i,j,1] = img1[i,j] / 255.0
            img1_processed[i,j,2] = img1[i,j] / 255.0
            img1_processed[i,j,0] = img1[i,j] / 255.0


cv2.imshow('Original', img1)
cv2.imshow('processed_image', img1_processed)
cv2.imwrite('processed_img_5-8.png', (img1_processed*255).astype(np.uint8))
cv2.waitKey()

#9. Color Image Enhancement by Histogram Processing 
img2 = cv2.imread('nchu_1112_4198_dip_hw_05_figs\Fig0635.tif', cv2.IMREAD_COLOR)

B, G, R = cv2.split(img2)
B_eq = cv2.equalizeHist(B)
G_eq = cv2.equalizeHist(G)
R_eq = cv2.equalizeHist(R)
img_eq = np.zeros_like(img2, dtype=np.uint8)
img_eq [:,:,0] = B_eq
img_eq [:,:,1] = G_eq
img_eq [:,:,2] = R_eq

cv2.imwrite('RGB_histogram_equl.tif', img_eq)

B_hist = cv2.calcHist([B], [0], None, [256], [0, 256])
G_hist = cv2.calcHist([G], [0], None, [256], [0, 256])
R_hist = cv2.calcHist([R], [0], None, [256], [0, 256])

avg_hist = (B_hist + G_hist + R_hist) / 3
avg_hist_normalized = cv2.normalize(avg_hist, None, 0, 255, cv2.NORM_MINMAX)

cdf = np.cumsum(avg_hist)
cdf = (cdf *255 / cdf[-1]).astype(np.uint8)

B_eq2 = cdf[B]
G_eq2 = cdf[G]
R_eq2 = cdf[R]
img_eq2 = np.zeros_like(img2, dtype=np.uint8)
img_eq2 [:,:,0] = B_eq2
img_eq2 [:,:,1] = G_eq2
img_eq2 [:,:,2] = R_eq2

cv2.imwrite('avg_histogram_equl.tif', img_eq2)

#10. Color Image Segmentation
img3 = plt.imread('nchu_1112_4198_dip_hw_05_figs\Fig0628.tif')
plt.subplot(1,2,1),plt.imshow(img3)

avg = np.mean(img3[385:405, 115:135], axis=(0, 1)).astype(np.uint8)
std_dev = np.std(img3[385:405, 115:135], axis=(0, 1))

M, N ,P= img3.shape
result = np.zeros_like(img3)
for i in range(M):
      for j in range(N):
            if (img3[i,j,0] >= avg[0]-1.25 * std_dev[0]) & (img3[i,j,0] < avg[0]+1.25 * std_dev[0]):
                if (img3[i,j,1] >= avg[1]-1.25 * std_dev[1]) & (img3[i,j,1] < avg[1]+1.25 * std_dev[1]):
                    if (img3[i,j,2] >= avg[2]-1.25 * std_dev[2]) & (img3[i,j,2] < avg[2]+1.25 * std_dev[2]):
                        result[i,j,0] = 255
                        result[i,j,1] = 255
                        result[i,j,2] = 255

plt.subplot(1,2,2),plt.imshow(result)
plt.show()
