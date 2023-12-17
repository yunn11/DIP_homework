#4109064030 陳沛昀
import cv2
import numpy as np
import math

img1 = cv2.imread('Fig0221.tif', cv2.IMREAD_GRAYSCALE)
intensity = int(input('Enter the intensity:'))
def quantizing (img, intensity):
    height, width = img.shape[0], img.shape[1]
    damn = int(math.log(intensity, 2))
    levels = 8-(damn)
    quantizing=np.zeros((height, width),dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            quantizing[i][j]=(img[i][j]>>levels)*(int(255/((intensity)-1)))
    return quantizing
quantized_image = quantizing(img1, intensity)
cv2.imshow ('reduce intensity', quantized_image)
#cv2.imwrite ('level_128.jpg', quantized_image)


img2 = cv2.imread('Fig0220.tif')
factor = int(input('Enter the factor:'))
width, height = img2.shape[1], img2.shape[0]
shrink_img = cv2.resize(img2, (int(width/factor), int(height/factor)), interpolation=cv2.INTER_NEAREST)
zoom_img = cv2.resize(shrink_img, (int(width), int(height)), interpolation=cv2.INTER_NEAREST)
cv2.imshow ('shrink', shrink_img)
cv2.imshow ('zoom', zoom_img)
cv2.imwrite ('shrink_image.jpg', shrink_img)
cv2.imwrite ('zoom_image.jpg', zoom_img)


img3 = cv2.imread('Fig0236.tif')
width, height = img3.shape[1], img3.shape[0]
rotation_degree = np.int8(input('Enter the rotation degree:'))
center = input('Enter the center:')
center_list = center.split(",")
my_tuple = tuple(map(int, center_list))
scale_ratio = np.float32(input('Enter the scale ratio:'))
transform_matrix = cv2.getRotationMatrix2D(my_tuple, rotation_degree, scale_ratio)
transform_img_1= cv2.warpAffine(img3, transform_matrix, (width, height), flags=cv2.INTER_NEAREST)
transform_img_2= cv2.warpAffine(img3, transform_matrix, (width, height), flags=cv2.INTER_AREA)
transform_img_3= cv2.warpAffine(img3, transform_matrix, (width, height), flags=cv2.INTER_CUBIC)
cv2.imshow('Nereast interpolation', transform_img_1)
cv2.imshow('Bilinear interpolation', transform_img_2)
cv2.imshow('Bicubic interpolation', transform_img_3)
#cv2.imwrite('transform_image_Bicubic.jpg', transform_img_1)

cv2.waitKey(0)
