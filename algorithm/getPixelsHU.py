import numpy as np
import pylibjpeg

def get_pixels_hu(scans):
    # Stack pixel arrays — float32 is sufficient and ~2× faster than float64
    image = np.stack([s.pixel_array for s in scans]).astype(np.float32)

    # Replace out-of-scan sentinel (-2000) with air (0 HU after conversion)
    image[image == -2000] = 0

    # Apply rescale slope + intercept in one vectorised pass
    slope     = float(scans[0].RescaleSlope)
    intercept = float(scans[0].RescaleIntercept)
    image = image * slope + intercept

    return image.astype(np.int16)
