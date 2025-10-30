#!/usr/bin/env python3
"""
compute_eta.py

Usage: edit the folder paths below if needed and run:
    python compute_eta.py
"""

import os
import glob
import numpy as np
from astropy.io import fits
import random
import sys

# === USER PATHS (edit if different) ===
u_dir = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\Oct01\oct01_2025\target2\u\high\keep"
clear_dir = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\Oct01\oct01_2025\target2\clear\high\keep"

# === SETTINGS ===
box_size = 40
n_boxes = 10
random_seed = 42  # reproducible selection

# === Helper functions ===
COMMON_EXPTIME_KEYS = ["EXPTIME", "EXPOSURE", "EXPTIM", "EXPT", "EXPOS", "EXPTIME_S", "EXPTIMS"]

def read_and_normalize(path_list):
    """
    Read FITS files, normalize by exposure time (value in header is in units of 1e-5 s),
    return a 3D numpy array shaped (N, ny, nx).
    """
    imgs = []
    for p in path_list:
        try:
            with fits.open(p, memmap=False) as hdul:
                # find first image-like HDU with data
                hdu = None
                for h in hdul:
                    if h.data is not None:
                        hdu = h
                        break
                if hdu is None:
                    print(f"Skipping {p}: no image data found", file=sys.stderr)
                    continue
                data = np.array(hdu.data, dtype=float)
                header = hdu.header

                # find exposure time key
                exptime_raw = None
                for k in COMMON_EXPTIME_KEYS:
                    if k in header:
                        exptime_raw = header[k]
                        break
                if exptime_raw is None:
                    # try lowercase or 'EXPTIME' variations
                    for k in header.keys():
                        if k.upper().startswith("EXPT"):
                            exptime_raw = header[k]
                            break
                if exptime_raw is None:
                    raise KeyError(f"Exposure time keyword not found in header for {p}")

                # exptime_raw is in units of 10**(-5) seconds; convert to seconds
                exptime_seconds = float(exptime_raw) * 1e-5
                if exptime_seconds <= 0:
                    raise ValueError(f"Non-positive exposure time in {p}: {exptime_seconds} s")

                data_norm = data / exptime_seconds
                # replace NaNs/Infs with 0 to avoid contaminating median
                data_norm = np.nan_to_num(data_norm, nan=0.0, posinf=0.0, neginf=0.0)
                imgs.append(data_norm)
        except Exception as e:
            print(f"Warning: failed to read {p}: {e}", file=sys.stderr)
    if len(imgs) == 0:
        raise RuntimeError("No valid normalized images were read from given paths.")
    imgs = np.stack(imgs, axis=0)  # shape (N, ny, nx)
    return imgs

def median_master(img_stack):
    """Return median frame across stack axis=0."""
    return np.median(img_stack, axis=0)

# === Collect file lists ===
u_files = sorted(glob.glob(os.path.join(u_dir, "*.fits")) + glob.glob(os.path.join(u_dir, "*.fit")))
clear_files = sorted(glob.glob(os.path.join(clear_dir, "*.fits")) + glob.glob(os.path.join(clear_dir, "*.fit")))

if len(u_files) == 0:
    raise SystemExit(f"No U files found in {u_dir}")
if len(clear_files) == 0:
    raise SystemExit(f"No Clear files found in {clear_dir}")

print(f"Found {len(u_files)} U files and {len(clear_files)} Clear files.")

# === Read and normalize ===
u_stack = read_and_normalize(u_files)
clear_stack = read_and_normalize(clear_files)

# === Check shapes match ===
if u_stack.shape[1:] != clear_stack.shape[1:]:
    # If shapes differ, attempt to crop to common intersection
    ny_u, nx_u = u_stack.shape[1], u_stack.shape[2]
    ny_c, nx_c = clear_stack.shape[1], clear_stack.shape[2]
    ny = min(ny_u, ny_c)
    nx = min(nx_u, nx_c)
    print(f"Warning: image shapes differ; cropping to common shape ({ny},{nx}).")
    u_stack = u_stack[:, :ny, :nx]
    clear_stack = clear_stack[:, :ny, :nx]

ny, nx = u_stack.shape[1], u_stack.shape[2]
if ny < box_size or nx < box_size:
    raise SystemExit(f"Images too small for box size {box_size}: image shape ({ny},{nx})")

# === Make masters ===
u_master = median_master(u_stack)
clear_master = median_master(clear_stack)

# === Choose random boxes inside u_master ===
random.seed(random_seed)
boxes = []
max_y0 = ny - box_size
max_x0 = nx - box_size

attempts = 0
while len(boxes) < n_boxes and attempts < n_boxes * 20:
    attempts += 1
    y0 = random.randint(0, max_y0)
    x0 = random.randint(0, max_x0)
    # ensure we don't pick duplicate boxes (same coords)
    if (y0, x0) in boxes:
        continue
    boxes.append((y0, x0))

if len(boxes) < n_boxes:
    print(f"Warning: only selected {len(boxes)} boxes (requested {n_boxes}).")

# === Compute sums and ratios ===
ratios = []
details = []
for idx, (y0, x0) in enumerate(boxes, start=1):
    y1 = y0 + box_size
    x1 = x0 + box_size
    u_sum = np.sum(u_master[y0:y1, x0:x1])
    c_sum = np.sum(clear_master[y0:y1, x0:x1])
    if c_sum == 0:
        ratio = np.nan
        print(f"Box {idx} at ({y0},{x0}): clear sum is zero; ratio set to NaN", file=sys.stderr)
    else:
        ratio = u_sum / c_sum
    ratios.append(ratio)
    details.append({
        "box_index": idx,
        "y0": y0,
        "x0": x0,
        "u_sum": float(u_sum),
        "clear_sum": float(c_sum),
        "ratio": float(ratio) if not np.isnan(ratio) else np.nan
    })

# filter out NaNs for the median
valid_ratios = [r for r in ratios if not np.isnan(r)]
if len(valid_ratios) == 0:
    eta = np.nan
else:
    eta = float(np.median(valid_ratios))

# === Print results ===
print("\nSelected boxes (y0, x0) and results:")
for d in details:
    print(f"Box {d['box_index']:2d}: y0={d['y0']:4d}, x0={d['x0']:4d}, "
          f"U_sum={d['u_sum']:12.3f}, Clear_sum={d['clear_sum']:12.3f}, "
          f"ratio={d['ratio']!s}")

print(f"\nMedian eta (median of valid Iu/Iclear ratios): {eta}")

# === Optionally save results to a small text file ===
out_txt = "eta_results.txt"
with open(out_txt, "w") as f:
    f.write("box_index,y0,x0,u_sum,clear_sum,ratio\n")
    for d in details:
        f.write(f"{d['box_index']},{d['y0']},{d['x0']},{d['u_sum']},{d['clear_sum']},{d['ratio']}\n")
    f.write(f"median_eta,{eta}\n")
print(f"Results saved to {out_txt}")