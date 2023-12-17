import cv2
import numpy as np
import matplotlib.pyplot as plt

#Histogram equalization
img1 = cv2.imread('nchu_1112_4198_dip_hw_05_figs\Fig0628.tif', cv2.IMREAD_GRAYSCALE)

hist,bins = np.histogram(img1.flatten(),256,[0,256])

plt.hist(img1.flatten(),256,[0,256], color = '#ff7389')
plt.xlim([0,256])
plt.show()


def equalized (img):
    width=img.shape[1]
    height=img.shape[0]
    J = np.copy(img)
    r = np.zeros(256,dtype=np.int8)
    s = np.zeros(256,dtype=np.int8)
    for i in range (height):
        for j in range (width):
            r [img[i][j]] += 1
    sum = 0
    for i in range (256):
        sum = sum+r[i]
        s[i] = int(255*sum/(width*height))
    for i in range (0,height):
        for j in range (0,width):
            index = img[i][j]
            J[i][j] = s[index]
    return J

equalization = equalized(img1)
plt.hist(equalization.flatten(),256,[0,255], color = '#41896c')
plt.show()
plt.imshow(equalization, cmap='gray')
plt.show()

#Enhancement Using the Laplacian
img2 = cv2.imread('nchu_1112_4198_dip_hw_02_figs\Fig0338(a)(blurry_moon).tif', cv2.IMREAD_GRAYSCALE)

def laplacian(image):
    n = int(input("Enter kernel size (odd integer): "))
    print(f"Enter {n}x{n} kernel elements, separated by spaces:")
    kernel = []
    for i in range(n):
        kernel.append(list(map(float, input().rstrip().split())))
    kernel = np.array(kernel, dtype=np.float32)
    laplacian_img = cv2.filter2D(image, -1, kernel)       
    return laplacian_img
print('輸入Laplacian kernel')
laplacian_img = laplacian(img2)
sharpen_img=img2+(-1)*laplacian_img
plt.subplot(131),plt.imshow(img2, cmap='gray'), plt.title('original')
plt.subplot(132),plt.imshow(laplacian_img, cmap='gray'), plt.title('mask')
plt.subplot(133),plt.imshow(sharpen_img, cmap='gray'), plt.title('laplacian')
plt.show()

#unsharp masking & highboost filter
img3 = cv2.imread('nchu_1112_4198_dip_hw_02_figs\Fig0340(a)(dipxe_text).tif', cv2.IMREAD_GRAYSCALE)
print('輸入averaging kernel')
blurred_img = laplacian(img3)
mask_img = img3 + (-1)*blurred_img
result = img3 + (4.5)*mask_img
plt.subplot(131),plt.imshow(img3, cmap='gray'), plt.title('original')
plt.subplot(132),plt.imshow(mask_img, cmap='gray'), plt.title('mask')
plt.subplot(133),plt.imshow(result, cmap='gray'), plt.title('highboost filtering k=4.5')
plt.show()

