import numpy as np

def thrSegmentation(img, min_HU, max_HU):
    """Return binary mask of pixels within [min_HU, max_HU]."""
    return ((img >= min_HU) & (img <= max_HU)).astype(np.uint8)


def remove_calcification(img, max_HU):
    mask = img < max_HU
    zeros = np.zeros_like(img)
    zeros[mask] = img[mask]
    return zeros


def applyMask(img, mask):
    import cv2 as cv
    return cv.bitwise_and(img, img, mask=mask)
