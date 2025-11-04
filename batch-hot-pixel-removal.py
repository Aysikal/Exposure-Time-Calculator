import numpy as np
import pywt
from astropy.io import fits
import matplotlib.pyplot as plt
import os
import csv
from datetime import datetime
from scipy.ndimage import median_filter
from astropy.visualization import ZScaleInterval, ImageNormalize

# === CONFIGURATION ===
num_iterations = 3
wavelet = 'db2'
dwt_level = 1
shifts = [(0, 0), (1, 0), (0, 1), (1, 1)]
fits_folder = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\rezaei_saba_farideH_2025_10_22\flat\i\high"
box_center = (1200, 1200)
star_center = (858, 2154)
box_size = 100
median_size = 5
hot_sigma = 6.0
protect_above = None
# === Ask for output folder ===
output_root = input("Enter output folder path: ").strip()

# Create subfolders
fits_out_dir = os.path.join(output_root, "hot pixels removed")
csv_out_dir = os.path.join(output_root, "hot pixel removal csv")
os.makedirs(fits_out_dir, exist_ok=True)
os.makedirs(csv_out_dir, exist_ok=True)


def extract_box(img, center, size):
    x, y = center
    half = size // 2
    return img[y-half:y+half, x-half:x+half].copy()

def stats(arr):
    arr_flat = arr[np.isfinite(arr)].ravel()
    if arr_flat.size == 0:
        return {
            'mean': np.nan,
            'median': np.nan,
            'std': np.nan,
            'rms': np.nan,
            'min': np.nan,
            'max': np.nan
        }
    return {
        'mean': float(np.nanmean(arr_flat)),
        'median': float(np.nanmedian(arr_flat)),
        'std': float(np.nanstd(arr_flat)),
        'rms': float(np.sqrt(np.nanmean(arr_flat**2))),
        'min': float(np.nanmin(arr_flat)),
        'max': float(np.nanmax(arr_flat))
    }

def detect_hot_mask(img, median_size=5, hot_sigma=6.0, protect_above=None):
    med = median_filter(img, size=median_size, mode='reflect')
    resid = img - med
    sigma = np.nanstd(resid)
    if sigma <= 0:
        return np.zeros_like(img, dtype=bool), med
    mask = np.abs(resid) > (hot_sigma * sigma)
    if protect_above is not None:
        mask = mask & (img < protect_above)
    return mask, med

def zero_finest_details_wavedec2(rolled, wavelet, level):
    coeffs = pywt.wavedec2(rolled, wavelet, level=level, mode='symmetric')
    if len(coeffs) > 1:
        coeffs_mod = list(coeffs)
        coeffs_mod[1] = tuple(np.zeros_like(arr) for arr in coeffs_mod[1])
        reco = pywt.waverec2(coeffs_mod, wavelet, mode='symmetric')
    else:
        reco = rolled.copy()
    return reco

def cycle_spin_clean(image, wavelet, level, shifts):
    img = image.astype(np.float32)
    accum = np.zeros_like(img, dtype=np.float64)
    for dx, dy in shifts:
        rolled = np.roll(np.roll(img, dx, axis=1), dy, axis=0)
        reco = zero_finest_details_wavedec2(rolled, wavelet, level)
        reco_unrolled = np.roll(np.roll(reco, -dx, axis=1), -dy, axis=0)
        reco_unrolled = reco_unrolled[:img.shape[0], :img.shape[1]]
        accum += reco_unrolled
    cleaned = (accum / len(shifts)).astype(np.float32)
    return cleaned

# === PROCESS ALL FITS FILES IN FOLDER ===
for fname in os.listdir(fits_folder):
    if not fname.lower().endswith(('.fit', '.fits')):
        continue

    out_prefix = os.path.splitext(fname)[0] + "_cycleclean"
    diagnostics_csv = os.path.join(csv_out_dir, f"{out_prefix}_diagnostics.csv")
    out_fits = os.path.join(fits_out_dir, f"{out_prefix}_iter{num_iterations}.fit")
    png_out = os.path.join(csv_out_dir, f"{out_prefix}_orig_vs_final_zscale.png")

    fits_path = os.path.join(fits_folder, fname)
    hdul = fits.open(fits_path)
    data = hdul[0].data.astype(np.float32)
    hdr = hdul[0].header
    hdul.close()

    if data.ndim != 2:
        print(f"Skipping non-2D image: {fname}")
        continue

    orig_stats = stats(data)
    current_image = data.copy()
    iteration_stats = []

    for i in range(1, num_iterations + 1):
        mask_hot_start, med = detect_hot_mask(current_image, median_size, hot_sigma, protect_above)
        num_detected_start = int(np.count_nonzero(mask_hot_start))
        num_replaced = 0

        if np.any(mask_hot_start):
            inpainted = current_image.copy()
            inpainted[mask_hot_start] = med[mask_hot_start]
            cleaned_full = cycle_spin_clean(inpainted, wavelet, dwt_level, shifts)
            changed_mask = mask_hot_start & (cleaned_full != current_image)
            num_replaced = int(np.count_nonzero(changed_mask))
            current_image[mask_hot_start] = cleaned_full[mask_hot_start]
        else:
            cleaned_full = cycle_spin_clean(current_image, wavelet, dwt_level, shifts)
            changed_mask = (cleaned_full != current_image)
            num_replaced = int(np.count_nonzero(changed_mask))
            current_image[:] = cleaned_full

        current_image = np.clip(current_image, np.min(data), np.max(data))
        post_mask, _ = detect_hot_mask(current_image, median_size, hot_sigma, protect_above)
        num_detected_post = int(np.count_nonzero(post_mask))

        s_clean_whole = stats(current_image)
        s_clean_box = stats(extract_box(current_image, box_center, box_size))
        s_clean_star = stats(extract_box(current_image, star_center, box_size))
        percent_rms_reduction = 100.0 * (1.0 - s_clean_whole['rms'] / orig_stats['rms']) if orig_stats['rms'] != 0 else 0.0

        iteration_stats.append({
            'label': f"Iter {i}",
            'hot_pixels_after': num_detected_post,
            'hot_pixels_start': num_detected_start,
            'hot_pixels_replaced': num_replaced,
            'mean_whole': s_clean_whole['mean'],
            'std_whole': s_clean_whole['std'],
            'rms_whole': s_clean_whole['rms'],
            'mean_box': s_clean_box['mean'],
            'std_box': s_clean_box['std'],
            'rms_box': s_clean_box['rms'],
            'mean_star': s_clean_star['mean'],
            'std_star': s_clean_star['std'],
            'rms_star': s_clean_star['rms'],
            'percent_rms_reduction': percent_rms_reduction
        })

    # Save cleaned FITS
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_fits_path = os.path.join(fits_out_dir, f"{out_prefix}_iter{num_iterations}.fit")
    fits.PrimaryHDU(current_image, header=hdr).writeto(out_fits_path, overwrite=True)

    # Save diagnostics CSV
    with open(diagnostics_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(iteration_stats[0].keys()))
        writer.writeheader()
        for row in iteration_stats:
            writer.writerow(row)

    # Save ZScale comparison
    z = ZScaleInterval()
    norm_orig = ImageNormalize(data, interval=z)
    norm_final = ImageNormalize(current_image, interval=z)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    axes[0].imshow(data, origin='lower', cmap='gray_r', norm=norm_orig)
    axes[0].set_title('Original'); axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[1].imshow(current_image, origin='lower', cmap='gray_r', norm=norm_final)
    axes[1].set_title('Final cleaned'); axes[1].set_xticks([]); axes[1].set_yticks([])

    cbar = fig.colorbar(axes[1].images[0], ax=axes.ravel().tolist(), shrink=0.8, pad=0.02)
    cbar.set_label('DN')

    png_out_path = os.path.join(csv_out_dir, f"{out_prefix}_orig_vs_final_zscale.png")
    plt.savefig(png_out_path, dpi=200)
    plt.close()

    print(f"✔ Processed: {fname}")
    print(f"  ↪ Cleaned FITS saved to: {out_fits_path}")
    print(f"  ↪ Diagnostics CSV saved to: {diagnostics_csv}")
    print(f"  ↪ ZScale PNG saved to: {png_out_path}")
