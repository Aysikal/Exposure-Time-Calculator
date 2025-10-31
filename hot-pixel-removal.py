# cycle_spinning_table_original_firstcol_postcount_fixed.py
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
fits_path = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\rezaei_saba_farideH_2025_10_22\ngc869\r\high\keep\ngc891_r_2025_10_22_1x1_exp00.02.00.000_000001_High_1.fit"
box_center = (1200, 1200)
star_center = (858, 2154)
box_size = 100
save_cleaned_fits = True
out_prefix = "ngc891_cycleclean"
diagnostics_csv = f"{out_prefix}_diagnostics.csv"
stats_summary_txt = f"{out_prefix}_summary.txt"

# Hot-pixel detection / inpaint params
median_size = 5
hot_sigma = 6.0
protect_above = None

# === HELPERS ===
def extract_box(img, center, size):
    x, y = center
    half = size // 2
    return img[y-half:y+half, x-half:x+half].copy()

def stats(arr):
    arr_flat = arr[np.isfinite(arr)].ravel()
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

# === LOAD IMAGE ===
hdul = fits.open(fits_path)
data = hdul[0].data.astype(np.float32)
hdr = hdul[0].header
hdul.close()

if data.ndim != 2:
    raise ValueError("Expected a 2D image in primary HDU")

# compute original mask/stats
orig_mask, orig_med = detect_hot_mask(data, median_size=median_size, hot_sigma=hot_sigma, protect_above=protect_above)
orig_hot_count = int(np.count_nonzero(orig_mask))
orig_stats = stats(data)
orig_box = extract_box(data, box_center, box_size)
orig_box_stats = stats(orig_box)
orig_star_box = extract_box(data, star_center, box_size)
orig_star_stats = stats(orig_star_box)

# Prepare containers
current_image = data.copy()
iteration_stats = []
images_per_iter = [data.copy()]  # snapshots for plotting

# === ITERATIONS ===
for i in range(1, num_iterations + 1):
    mask_hot_start, med = detect_hot_mask(current_image, median_size=median_size, hot_sigma=hot_sigma, protect_above=protect_above)
    num_detected_start = int(np.count_nonzero(mask_hot_start))
    num_replaced = 0

    if np.any(mask_hot_start):
        inpainted = current_image.copy()
        inpainted[mask_hot_start] = med[mask_hot_start]
        cleaned_full = cycle_spin_clean(inpainted, wavelet, dwt_level, shifts)
        cleaned_full = cleaned_full[:data.shape[0], :data.shape[1]]
        changed_mask = mask_hot_start & (cleaned_full != current_image)
        num_replaced = int(np.count_nonzero(changed_mask))
        current_image[mask_hot_start] = cleaned_full[mask_hot_start]
    else:
        cleaned_full = cycle_spin_clean(current_image, wavelet, dwt_level, shifts)
        cleaned_full = cleaned_full[:data.shape[0], :data.shape[1]]
        changed_mask = (cleaned_full != current_image)
        num_replaced = int(np.count_nonzero(changed_mask))
        current_image[:] = cleaned_full

    current_image = np.clip(current_image, np.min(data), np.max(data))

    post_mask, post_med = detect_hot_mask(current_image, median_size=median_size, hot_sigma=hot_sigma, protect_above=protect_above)
    num_detected_post = int(np.count_nonzero(post_mask))

    images_per_iter.append(current_image.copy())

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
if save_cleaned_fits:
    cleaned_hdu = fits.PrimaryHDU(current_image, header=hdr)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_fits = f"{out_prefix}_iter{num_iterations}_{timestamp}.fit"
    cleaned_hdu.writeto(out_fits, overwrite=True)

# Write iteration stats CSV
fieldnames = ['label','hot_pixels_after','hot_pixels_start','hot_pixels_replaced',
              'mean_whole','std_whole','rms_whole','mean_box','std_box','rms_box',
              'mean_star','std_star','rms_star','percent_rms_reduction']
with open(diagnostics_csv, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in iteration_stats:
        writer.writerow(row)

# === 3D SURFACES: background and star region ===
X, Y = np.meshgrid(np.arange(box_size), np.arange(box_size))
vmin = np.percentile(data, 1)
vmax = np.nanmax(data)

# Background boxes per iteration
boxes_bg = [extract_box(img, box_center, box_size) for img in images_per_iter]
# Star boxes per iteration
boxes_star = [extract_box(img, star_center, box_size) for img in images_per_iter]

# Plot BG
fig_bg = plt.figure(figsize=(4*(num_iterations+1), 4))
for i, box in enumerate(boxes_bg):
    ax = fig_bg.add_subplot(1, num_iterations+1, 1+i, projection='3d')
    surf = ax.plot_surface(X, Y, box, cmap='viridis', edgecolor='none',
                           vmin=vmin, vmax=vmax, rcount=box_size, ccount=box_size)
    ax.set_title(f"BG Iter {i}")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_zlim(vmin, vmax)
    fig_bg.colorbar(surf, ax=ax, shrink=0.6, pad=0.05, label='DN')
plt.tight_layout()
plt.savefig(f"{out_prefix}_3d_bg.png", dpi=150)
plt.show()

fig_star = plt.figure(figsize=(4*(num_iterations+1), 6))
for i, box in enumerate(boxes_star):
    ax = fig_star.add_subplot(1, num_iterations+1, 1+i, projection='3d')
    surf = ax.plot_surface(X, Y, box, cmap='viridis', edgecolor='none',
                           vmin=vmin, vmax=vmax, rcount=box_size, ccount=box_size)
    ax.set_title(f"Star Iter {i}")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_zlim(vmin, vmax)
    ax.view_init(elev=30, azim=-60)  # adjust as needed
    fig_star.colorbar(surf, ax=ax, shrink=0.6, pad=0.05, label='DN')

plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.3)
plt.savefig(f"{out_prefix}_3d_star.png", dpi=150)
plt.show()


# Show side-by-side original vs final (ZScale)
orig = data.astype(np.float32)
final = current_image.astype(np.float32)
z = ZScaleInterval()
norm_orig = ImageNormalize(orig, interval=z)
norm_final = ImageNormalize(final, interval=z)

fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
axes[0].imshow(orig, origin='lower', cmap='gray_r', norm=norm_orig)
axes[0].set_title('Original'); axes[0].set_xticks([]); axes[0].set_yticks([])
axes[1].imshow(final, origin='lower', cmap='gray_r', norm=norm_final)
axes[1].set_title('Final cleaned'); axes[1].set_xticks([]); axes[1].set_yticks([])

# Shared colorbar
cbar = fig.colorbar(axes[1].images[0], ax=axes.ravel().tolist(), shrink=0.8, pad=0.02)
cbar.set_label('DN')

# Save PNG
png_out = f"{out_prefix}_orig_vs_final_zscale.png"
plt.savefig(png_out, dpi=200)
plt.show()

print(f"Saved comparison image to: {os.path.abspath(png_out)}")
print(f"Saved cleaned FITS to: {os.path.abspath(out_fits)}")
print(f"Saved diagnostics CSV to: {os.path.abspath(diagnostics_csv)}")
