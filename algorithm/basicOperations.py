import cv2 as cv
import numpy as np
from skimage.filters.thresholding import threshold_otsu

'@param image: dtype different of np.uint8'
'@return 3-channel BGR image'
def convert_image_to_rgb(image, filename, output_folder):
    # Normalize to [0, 255] — same behaviour as plt.imsave(..., cmap='gray')
    # but avoids the matplotlib backend overhead and double disk I/O.
    img = np.asarray(image, dtype=np.float32)
    vmin, vmax = float(img.min()), float(img.max())
    if vmax > vmin:
        img = ((img - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
    else:
        img = np.zeros(img.shape, dtype=np.uint8)
    img_bgr = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.imwrite(f'{output_folder}{filename}.png', img_bgr)
    return img_bgr

'@param mask: numpy.ndarray with values 0 (non-fat pixels) and 1 (fat pixels)'
'@param img: 3-channel image'
'@return 3-channel image mask+img'
def add_images(mask, img):
    # Direct boolean indexing — avoids dstack + np.where overhead
    img[mask != 0] = [172, 16, 12]
    return img

'@param img: 3-channel image'
'@return binary image'
def otsu_mask(img):
    thr = threshold_otsu(img)
    binary = img > thr
    return binary

'@param img: 3-channel image'
'@param mask: numpy.ndarray with values 0 and 1'
'@return binary image'
def connect_components(binary):
    new_img = np.zeros_like(binary)
    for val in np.unique(binary)[1:]:
        mask = np.uint8(binary == val)
        labels, stats = cv.connectedComponentsWithStats(mask, 4)[1:3]
        largest_label = 1 + np.argmax(stats[1:, cv.CC_STAT_AREA])
        new_img[labels == largest_label] = val
    return new_img

'@param img: 3-channel image'
'@param mask: numpy.ndarray with values 0 and 1'
'@return binary image'
def segmentation(img, mask):
    zeros = np.zeros_like(img)-300
    zeros[mask] = img[mask]
    return zeros

'@param hull: binary (boolean) image'
def opening(hull):
    hull = np.uint8(hull)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (40, 40))
    opening = cv.morphologyEx(hull, cv.MORPH_OPEN, kernel, iterations=3)
    return opening
