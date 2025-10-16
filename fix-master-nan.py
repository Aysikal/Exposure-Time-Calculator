import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip
from scipy.ndimage import median_filter
import os

# --- CONFIG ---
INPUT_FILE = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\masterframes\Oct 1 masterdarks\OLD\masterdark_58.13953s_2025-10-01_bin2x2_HIGH.fits"
OUTPUT_DIR = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\masterframes\Oct 1 masterdarks\HIGH"
SIGMA = 4
PRIMARY_SIZE = 3
FALLBACK_SIZE = 9

# --- Utility: replace NaNs using local median ---
def replace_nans_fast(img, primary_size=3, fallback_size=9):
    """
    Replace NaNs using local median filters. 
    If all neighbors are NaN, keep pixel as NaN.
    """
    nan_mask = np.isnan(img)
    if not np.any(nan_mask):
        return img, 0, 0

    # First pass (small window)
    median_small = median_filter(np.nan_to_num(img, nan=np.nanmedian(img)), size=primary_size, mode='mirror')
    filled = img.copy()
    primary_mask = nan_mask & ~np.isnan(median_small)
    filled[primary_mask] = median_small[primary_mask]

    # Second pass (larger window for remaining NaNs)
    still_nan = np.isnan(filled)
    if np.any(still_nan):
        median_large = median_filter(np.nan_to_num(img, nan=np.nanmedian(img)), size=fallback_size, mode='mirror')
        fallback_mask = still_nan & ~np.isnan(median_large)
        filled[fallback_mask] = median_large[fallback_mask]
    else:
        fallback_mask = np.zeros_like(img, dtype=bool)

    # Any pixel still NaN after both passes remains NaN
    return filled, np.sum(primary_mask), np.sum(fallback_mask)


# --- MAIN PROCESS ---
print("──────────────────────────────────────────────")
print("🔧 Processing master file:")
print(INPUT_FILE)
print("──────────────────────────────────────────────")

# Load FITS
with fits.open(INPUT_FILE) as hdul:
    data = hdul[0].data.astype(float)
    header = hdul[0].header

nan_count = np.isnan(data).sum()
print(f"🧮 Initial NaN count: {nan_count}")

if nan_count > 0:
    print(f"➡️ Step 1: Replacing NaNs using local median ({PRIMARY_SIZE}×{PRIMARY_SIZE}, fallback {FALLBACK_SIZE}×{FALLBACK_SIZE})...")
    data_filled, n_primary, n_fallback = replace_nans_fast(data, PRIMARY_SIZE, FALLBACK_SIZE)
    print(f"✅ Replaced {n_primary} NaNs (3x3) and {n_fallback} NaNs (fallback).")
else:
    print("✅ No NaNs found — skipping local median replacement.")
    data_filled = data

# --- Step 2: Sigma clipping ---
print(f"➡️ Step 2: Applying sigma clipping (σ = {SIGMA})...")
clipped = sigma_clip(data_filled, sigma=SIGMA, cenfunc='median')
data_clipped = clipped.filled(np.nan)

# --- Step 3: Fill NaNs again after sigma clip ---
nan_after_clip = np.isnan(data_clipped).sum()
print(f"🧮 NaNs after sigma clipping: {nan_after_clip}")

if nan_after_clip > 0:
    print(f"➡️ Filling sigma-clipped NaNs using local median ({PRIMARY_SIZE}×{PRIMARY_SIZE}, fallback {FALLBACK_SIZE}×{FALLBACK_SIZE})...")
    data_final, n_primary2, n_fallback2 = replace_nans_fast(data_clipped, PRIMARY_SIZE, FALLBACK_SIZE)
    print(f"✅ Replaced {n_primary2} NaNs (3x3) and {n_fallback2} NaNs (fallback).")
else:
    print("✅ No NaNs created by sigma clip — no further filling needed.")
    data_final = data_clipped

# --- Save output ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
basename = os.path.basename(INPUT_FILE).replace(".fits", "_fixed_sigma4.fits")
output_path = os.path.join(OUTPUT_DIR, basename)

hdr = header.copy()
# FITS headers must be ASCII — replace "σ" with "sigma"
hdr.add_history(f"NaN fixed + sigma clipped (sigma={SIGMA}) with local median replacement.")
fits.PrimaryHDU(data_final, hdr).writeto(output_path, overwrite=True)

print("──────────────────────────────────────────────")
print(f"✅ Fixed master saved to:\n{output_path}")
print("──────────────────────────────────────────────")
