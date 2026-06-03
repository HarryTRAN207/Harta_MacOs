import pydicom
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor

'Load the scans in given folder path'
'Scan = a set of dicom images from a certain patient'
def load_scan(path):
    # Filter out hidden files (.DS_Store, etc.) and non-DICOM entries
    files = [f for f in os.listdir(path)
             if not f.startswith('.') and os.path.isfile(os.path.join(path, f))]

    def _read(filename):
        return pydicom.dcmread(os.path.join(path, filename))

    # Parallel I/O — DICOM reading is I/O-bound; ThreadPoolExecutor is safe here
    with ThreadPoolExecutor(max_workers=min(8, len(files))) as executor:
        slices = list(executor.map(_read, files))

    slices.sort(key=lambda x: int(x.InstanceNumber))

    try:
        slice_thickness = np.abs(
            slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except Exception:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices
