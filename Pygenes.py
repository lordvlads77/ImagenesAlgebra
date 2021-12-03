import cv2
import numpy as np
from skimage import io, img_as_float

img_ajolote_noise = img_as_float(io.imread('Ajolote.jpg', as_gray=True))
img_lola_noise = img_as_float(io.imread('Lola.jpg', as_gray=True))
img_pez_noise = img_as_float(io.imread('Pez.jpg', as_gray=True))
img_chapala_noise = img_as_float(io.imread('chapala.jpg', as_gray=True))
img_chucky_noise = img_as_float(io.imread('Chucky.jpg', as_gray=True))
img_paloma_noise = img_as_float(io.imread('Paloma.jpg', as_gray=True))
img_serpiente_noise = img_as_float(io.imread('Serpiente.jpg', as_gray=True))

img_ajoloteGaussian = cv2.GaussianBlur(
    img_ajolote_noise, (5, 5), 1, borderType=cv2.BORDER_CONSTANT)
img_lola_gaussian = cv2.GaussianBlur(
    img_lola_noise, (5, 5), 1, borderType=cv2.BORDER_CONSTANT)
img_pez_gausssian = cv2.GaussianBlur(
    img_pez_noise, (5, 5), 1, borderType=cv2.BORDER_CONSTANT)
img_chapala_gaussian = cv2.GaussianBlur(
    img_chapala_noise, (5, 5), 30, borderType=cv2.BORDER_CONSTANT)
img_chucky_gaussian = cv2.GaussianBlur(
    img_chucky_noise, (5, 5), 30, borderType=cv2.BORDER_CONSTANT)
img_paloma_gaussian = cv2.GaussianBlur(
    img_paloma_noise, (5, 5), 30, borderType=cv2.BORDER_CONSTANT)
img_serpiente_gaussian = cv2.GaussianBlur(
    img_serpiente_noise, (5, 5), 5, borderType=cv2.BORDER_CONSTANT)

img_chapala_laplacian = cv2.Laplacian(
    img_chapala_gaussian, cv2.CV_64F, ksize=5)
img_chucky_laplacian = cv2.Laplacian(img_chucky_gaussian, cv2.CV_64F, ksize=5)
img_paloma_laplacian = cv2.Laplacian(img_paloma_gaussian, cv2.CV_64F, ksize=5)
img_serpiente_laplacian = cv2.Laplacian(
    img_serpiente_gaussian, cv2.CV_64F, ksize=5)

serpiente8b = np.uint8(img_serpiente_laplacian)

img_serpiente_canny = cv2.Canny(serpiente8b, 100, 200)


cv2.imshow("AjoloteGaussian", img_ajoloteGaussian)
cv2.imshow("LolaGaussian", img_lola_gaussian)
cv2.imshow("PezGaussian", img_pez_gausssian)
cv2.imshow("ChapalaLaplacian", img_chapala_laplacian)
cv2.imshow("ChuckyLaplacian", img_chucky_laplacian)
cv2.imshow("PalomaLaplacian", img_paloma_laplacian)
cv2.imshow("SerpienteLaplacian", img_serpiente_laplacian)
cv2.imshow("SerpienteCanny", serpiente8b)
cv2.waitKey(0)

#scale_percent = 50
#width = int(img_chapala_gaussian.shape[1] * scale_percent/100)
#height = int(img_chapala_gaussian.shape[0] * scale_percent/100)
#dim = (width, height)

#resized = cv2.resize(img_chapala_gaussian, dim, interpolation = cv2.INTER_AREA)
