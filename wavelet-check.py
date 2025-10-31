"""
Compare wavelet families (db1, db2) and decomposition levels (1–3)
on your FITS image cleaning workflow.
Includes background and star-region inspection.
"""

import numpy as np
import pywt
from astropy.io import fits
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt

# ==============================
# Configuration
# ==============================
fits_file = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\rezaei_saba_farideH_2025_10_22\ngc869\r\high\keep\ngc891_r_2025_10_22_1x1_exp00.02.00.000_000001_High_1.fit" # <-- change to your file
median_size = 5
hot_sigma = 6
wavelets = ["db1", "db2", "sym4"]   # test different wavelet families
levels = [1, 2, 3]                  # test different decomposition depths
cycle_shifts = [(0,0), (1,0), (0,1), (1,1)]  # 2x2 shifts (cycle spinning)
bg_center = (1200, 1200)
star_center = (858, 2154)
roi_size = 100  # pixels for cropped box (half-size 50 each way)

# ==============================
# Helper functions
# ==============================

def detect_hot_pixels(img, sigma_thresh):
    """Detect hot pixels relative to robust background statistics."""
    med = np.median(img)
    mad = 1.4826 * np.median(np.abs(img - med))
    mask = img > (med + sigma_thresh * mad)
    return mask

def replace_hot_pixels(img, mask, median_size):
    """Replace hot pixels with local median."""
    median_img = median_filter(img, size=median_size)
    out = img.copy()
    out[mask] = median_img[mask]
    return out

def zero_finest_details_wavedec2(img, wavelet, level):
    """Remove the finest detail coefficients (LH, HL, HH) at level 1."""
    coeffs = pywt.wavedec2(img, wavelet=wavelet, level=level, mode='symmetric')
    coeffs_mod = list(coeffs)
    # zero-out the *finest* detail (coeffs[-1]) not coeffs[1]
    coeffs_mod[-1] = tuple(np.zeros_like(arr) for arr in coeffs_mod[-1])
    reco = pywt.waverec2(coeffs_mod, wavelet=wavelet, mode='symmetric')
    return reco

def cycle_spin_clean(img, wavelet, level, shifts):
    """Cycle spinning to reduce shift artifacts."""
    img = img.astype(np.float32)
    accum = np.zeros_like(img, dtype=np.float64)
    for dx, dy in shifts:
        rolled = np.roll(np.roll(img, dx, axis=1), dy, axis=0)
        reco = zero_finest_details_wavedec2(rolled, wavelet, level)
        reco_unrolled = np.roll(np.roll(reco, -dx, axis=1), -dy, axis=0)
        accum += reco_unrolled
    cleaned = (accum / len(shifts)).astype(np.float32)
    return cleaned

def crop_center(img, center, size):
    x, y = center
    half = size // 2
    return img[y-half:y+half, x-half:x+half]

def rms(arr):
    return np.sqrt(np.mean(arr**2))

# ==============================
# Load image and preprocess
# ==============================
img = fits.getdata(fits_file).astype(np.float32)
print(f"Loaded image: {img.shape}, min={img.min():.2f}, max={img.max():.2f}")

# Initial hot pixel detection + replacement
mask_hot = detect_hot_pixels(img, hot_sigma)
img_replaced = replace_hot_pixels(img, mask_hot, median_size)
print(f"Hot pixels replaced: {np.sum(mask_hot)}")

# ==============================
# Run tests
# ==============================
results = {}
for wavelet in wavelets:
    for level in levels:
        print(f"Processing wavelet={wavelet}, level={level}...")
        cleaned = cycle_spin_clean(img_replaced, wavelet, level, cycle_shifts)
        results[(wavelet, level)] = cleaned

# ==============================
# Metrics and ROI comparison
# ==============================
fig, axs = plt.subplots(len(wavelets), len(levels), figsize=(14, 9))
plt.subplots_adjust(wspace=0.05, hspace=0.1)

bg_roi_orig = crop_center(img, bg_center, roi_size)
star_roi_orig = crop_center(img, star_center, roi_size)

print("\n=== Metrics ===")
print(f"{'Wavelet':<8} {'Level':<5} {'Whole_STD':>10} {'BG_STD':>10} {'BG_RMS':>10} {'Star_STD':>10} {'Star_RMS':>10}")
for i, wavelet in enumerate(wavelets):
    for j, level in enumerate(levels):
        cleaned = results[(wavelet, level)]
        bg_roi = crop_center(cleaned, bg_center, roi_size)
        star_roi = crop_center(cleaned, star_center, roi_size)
        std_whole = np.std(cleaned)
        std_bg = np.std(bg_roi)
        rms_bg = rms(bg_roi)
        std_star = np.std(star_roi)
        rms_star = rms(star_roi)
        print(f"{wavelet:<8} {level:<5d} {std_whole:10.3f} {std_bg:10.3f} {rms_bg:10.3f} {std_star:10.3f} {rms_star:10.3f}")

        axs[i,j].imshow(cleaned, origin='lower', cmap='gray', vmin=np.percentile(img,1), vmax=np.percentile(img,99))
        axs[i,j].set_title(f"{wavelet} L{level}")
        axs[i,j].axis('off')

plt.suptitle("db1 vs db2 vs sym4 — different DWT levels", fontsize=14)
plt.show()

# ==============================
# ROI visual comparison
# ==============================
fig2, axs2 = plt.subplots(len(wavelets), len(levels)*2, figsize=(16, 9))
plt.subplots_adjust(wspace=0.05, hspace=0.1)

for i, wavelet in enumerate(wavelets):
    for j, level in enumerate(levels):
        cleaned = results[(wavelet, level)]
        bg_roi = crop_center(cleaned, bg_center, roi_size)
        star_roi = crop_center(cleaned, star_center, roi_size)
        axs2[i, j*2].imshow(bg_roi, cmap='gray', origin='lower')
        axs2[i, j*2].set_title(f"{wavelet} L{level} — BG")
        axs2[i, j*2+1].imshow(star_roi, cmap='gray', origin='lower')
        axs2[i, j*2+1].set_title(f"{wavelet} L{level} — Star")
        for ax in (axs2[i, j*2], axs2[i, j*2+1]):
            ax.axis('off')

plt.suptitle("ROI comparison: background (left) vs star region (right)", fontsize=14)
plt.show()
