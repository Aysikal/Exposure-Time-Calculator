#!/usr/bin/env python3
"""
stack_aligned.py

Stack all aligned FITS in the folder (median or mean), preserve a representative header,
and write a final stacked FITS that viewers will show correctly.

Usage: edit INPUT_DIR and METHOD then run.
"""

from pathlib import Path
from glob import glob
import numpy as np
from astropy.io import fits
import logging

# CONFIGURE
INPUT_DIR = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\reduced\October 1st, area 95 green high\keep\aligned ۱"
OUTNAME = "stacked.fits"   # name of output file written into INPUT_DIR
METHOD = "median"          # "median" or "mean"
CLIP_SIGMA = None          # e.g., 3.0 to sigma-clip along stack axis, or None to skip clipping
# END CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

def list_fits(dirpath):
    p = Path(dirpath)
    patterns = ["*_aligned.fits", "*.fits", "*.fit", "*.fts"]
    files = []
    for pat in patterns:
        files.extend(sorted(glob(str(p / pat))))
    # prefer aligned files if present
    aligned = [f for f in files if "_aligned" in Path(f).name]
    return aligned if aligned else files

def load_primary_array(path):
    with fits.open(path, memmap=False) as hdul:
        # get first HDU with 2D numeric data
        for hdu in hdul:
            if hdu.data is None:
                continue
            arr = np.asarray(hdu.data)
            if arr.ndim >= 2:
                # if more dims, take first 2D plane
                if arr.ndim > 2:
                    arr = arr[0]
                arr = arr.astype(np.float64, copy=False)
                # convert masked arrays to numeric and replace inf/nan
                arr = np.nan_to_num(arr, nan=np.nan, posinf=np.nan, neginf=np.nan)
                return arr, hdul  # return array and full hdul for header preservation
    raise RuntimeError(f"No 2D image found in {path}")

def sigma_clip_stack(stack, sigma=3.0, iters=3):
    # stack shape: (N, y, x)
    data = stack.copy()
    mask = np.isnan(data)
    for _ in range(iters):
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        # expand mean/std to stack shape for comparison
        upper = mean + sigma * std
        lower = mean - sigma * std
        new_mask = (data > upper) | (data < lower)
        data[new_mask] = np.nan
        mask = mask | new_mask
    return data

def bitpix_for_dtype(dtype):
    if np.issubdtype(dtype, np.floating):
        return -64 if np.dtype(dtype) == np.float64 else -32
    if np.issubdtype(dtype, np.signedinteger):
        return 16 if np.dtype(dtype).itemsize == 2 else 32
    return -32

def main():
    files = list_fits(INPUT_DIR)
    if not files:
        logging.error("No FITS files found in %s", INPUT_DIR)
        return
    logging.info("Found %d files", len(files))

    arrays = []
    hdul_ref = None
    for i, f in enumerate(files):
        arr, hdul = load_primary_array(f)
        if i == 0:
            shape_ref = arr.shape
            hdul_ref = hdul
        else:
            if arr.shape != shape_ref:
                logging.warning("Skipping %s shape mismatch %s != %s", f, arr.shape, shape_ref)
                continue
        arrays.append(arr)

    if not arrays:
        logging.error("No valid images to stack after shape check.")
        return

    stack = np.stack(arrays, axis=0)  # shape (N, y, x)
    logging.info("Stack shape %s method=%s", stack.shape, METHOD)

    if CLIP_SIGMA is not None and CLIP_SIGMA > 0:
        logging.info("Applying sigma clipping sigma=%.2f", CLIP_SIGMA)
        stack = sigma_clip_stack(stack, sigma=CLIP_SIGMA, iters=3)

    if METHOD.lower() == "median":
        result = np.nanmedian(stack, axis=0)
    elif METHOD.lower() == "mean":
        result = np.nanmean(stack, axis=0)
    else:
        logging.error("Unknown METHOD %s. Use 'median' or 'mean'.", METHOD)
        return

    # If all values at a pixel are NaN, keep NaN; viewers may show as blank. Replace with 0 if desired.
    # result = np.nan_to_num(result, nan=0.0)

    # Prepare header: preserve hdul_ref but update primary structural keywords
    hdul_out = fits.HDUList([h.copy() for h in hdul_ref])
    # assign data (ensure 2D)
    arr_out = result
    if arr_out.ndim > 2:
        arr_out = arr_out[0]
    hdul_out[0].data = arr_out
    hdr = hdul_out[0].header

    # Update structural keywords
    hdr['NAXIS'] = 2
    hdr['NAXIS1'] = int(arr_out.shape[1])
    hdr['NAXIS2'] = int(arr_out.shape[0])
    hdr['BITPIX'] = bitpix_for_dtype(arr_out.dtype)

    hdr.add_history(f"Stacked {len(arrays)} frames method={METHOD} clip_sigma={CLIP_SIGMA}")

    outpath = Path(INPUT_DIR) / OUTNAME
    hdul_out.writeto(outpath, overwrite=True)
    logging.info("Wrote stacked file %s", outpath)

if __name__ == "__main__":
    main()
