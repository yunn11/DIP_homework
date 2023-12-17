import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift, ifft2

# Noise Reduction Using a Median Filter
def median_filter(image):
    filtered_image = np.zeros(image.shape, dtype=np.uint8)
    for i in range(1, image.shape[0]-1):
        for j in range(1, image.shape[1]-1):
            window = [
                image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                image[i, j-1], image[i, j], image[i, j+1],
                image[i+1, j-1], image[i+1, j], image[i+1, j+1]
            ]
            median = np.median(window)
            filtered_image[i, j] = median

    return filtered_image

image1 = cv2.imread('nchu_1112_4198_dip_hw_04_figs\Fig0507(a)(ckt-board-orig).tif', cv2.IMREAD_GRAYSCALE)
noisy_image = np.copy(image1)
pa = pb = 0.2
mask = np.random.choice((0, 1, 2), size=image1.shape, p=[1-pa-pb, pa, pb])
noisy_image[mask == 1] = 255
noisy_image[mask == 2] = 0

filtered_image = median_filter(noisy_image)

combined_image = np.hstack((image1, noisy_image, filtered_image))
cv2.imshow('Original | Noisy | Filtered', combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Periodic Noise Reduction Using a Notch Filter
def add_sinusoidal_noise(image, A, u0, v0):
    rows, cols = image.shape
    x = np.arange(cols)
    y = np.arange(rows)
    xx, yy = np.meshgrid(x,y)
    noise = A * np.sin(u0*xx/cols + v0*yy/rows)
    # Add the noise pattern to the image
    noisy_image = noise.astype(np.float32) + image.astype(np.float32)
    # Clip the pixel values to the range [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image

image2 = cv2.imread('nchu_1112_4198_dip_hw_04_figs\Fig0526(a)(original_DIP).tif', cv2.IMREAD_GRAYSCALE)
image2 = cv2.resize(image2, (256,256), interpolation=cv2.INTER_AREA)
M,N = image2.shape
noisy_image2 = add_sinusoidal_noise(image2, 100, M/2, 0)
plt.subplot(1,3,1),plt.imshow(noisy_image2, cmap='gray'), plt.title('sin noisy image')


ft = fft2(noisy_image2)
ft = fftshift(ft)
# ft = abs(ft)
notch_filter = np.zeros_like(noisy_image2)
notch_filter[:, :] = 0
width = int(input('Enter the width:'))
notch_filter[:, 128-width:128+width] = 1
magnitude_spectrum = (notch_filter) * ft
spectrum = np.abs(magnitude_spectrum)
spectrum = np.log(1 + spectrum)
plt.subplot(1,3,2),plt.imshow(spectrum, cmap='gray'), plt.title('sin_noisy_spectrum')

Forier = fftshift(magnitude_spectrum)
Forier = ifft2(Forier).real
Forier = np.abs(Forier)

print(Forier)
plt.subplot(1,3,3),plt.imshow(Forier, cmap='gray'), plt.title('restore')
plt.show()