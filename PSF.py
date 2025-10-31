#════════════════════════════════════════════════════════════════════════════════════════════════#
# This code was written by Aysan Hemmatiortakand. Last updated 9/30/2025 (patched)
# Contact: aysanhemmatiortakand@gmail.com  GitHub: https://github.com/Aysikal
# Behavior: SKIP any rows in the .npy where x or y is NaN (no recovery). Outputs saved.
#════════════════════════════════════════════════════════════════════════════════════════════════#

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import simple_norm
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

# ---------- User-editable settings ----------
box_size = 200
native_pixel_scale = 0.047 * 1.8    # arcsec / pixel at bin=1
binning = 1                         # set detector binning here (integer)
pixel_scale = native_pixel_scale * binning
color = "gray"
filter_name = "g"
mode = "High"
folder_path = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\rezaei_saba_farideH_2025_10_21\ngc604\g\low"
star_coordinates_loc = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\Star Coords\extiction\Oct 21\NGC 604\g\oct21-g-NGC604-star3.npy"
specific_plot_idx = 0
output_plots_dir = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\plots\PSF"
# ---------- End settings ----------

os.makedirs(output_plots_dir, exist_ok=True)

# ---------- helpers ----------
def open_fits(path):
    with fits.open(path) as fitsfile:
        return fitsfile[0].data.astype(float), fitsfile[0].header

def calculate_com(data):
    total = np.sum(data)
    if total == 0:
        h, w = data.shape
        return w/2.0, h/2.0
    y, x = np.indices(data.shape)
    com_y = np.sum(y * data) / total
    com_x = np.sum(x * data) / total
    return com_x, com_y

def calculate_radial_profile(data, center, max_radius):
    y, x = np.indices(data.shape)
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r_int = r.astype(int)
    mask = r_int <= max_radius
    if mask.sum() == 0:
        return np.array([])
    tbin = np.bincount(r_int[mask].ravel(), data[mask].ravel())
    nr = np.bincount(r_int[mask].ravel())
    radialprofile = np.zeros_like(tbin, dtype=float)
    radialprofile[nr > 0] = tbin[nr > 0] / nr[nr > 0]
    return gaussian_filter1d(radialprofile, sigma=2)

def gaussian(x, A, mu, sigma):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def find_hwhm(profile):
    if len(profile) < 3:
        return np.nan
    peak_idx = np.nanargmax(profile)
    peak_val = profile[peak_idx]
    tail_len = max(1, min(15, len(profile)//10))
    background = np.median(profile[-tail_len:]) if tail_len > 0 else 0.0
    half_max = (peak_val + background) / 2.0
    for r in range(peak_idx, len(profile)-1):
        if (profile[r] >= half_max >= profile[r+1]) or (profile[r] <= half_max <= profile[r+1]):
            denom = (profile[r] - profile[r+1])
            if denom == 0:
                return float(r)
            frac = (profile[r] - half_max) / denom
            return r + frac
    return np.nan

def fit_fwhm_from_profile(profile):
    r = np.arange(len(profile))
    mask = r <= (len(r)//2)
    if mask.sum() < 5:
        return 2.0 * find_hwhm(profile)
    p0 = [np.nanmax(profile), np.nanargmax(profile), 2.0]
    try:
        popt, pcov = curve_fit(gaussian, r[mask], profile[mask], p0=p0, maxfev=5000)
        sigma = popt[2]
        fwhm_pixels = 2.0 * np.sqrt(2.0 * np.log(2.0)) * abs(sigma)
        return fwhm_pixels / 2.0
    except Exception:
        return find_hwhm(profile)

# ---------- file lists ----------
image_files = [
    os.path.join(folder_path, f)
    for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.fit', '.fits'))
]
image_files.sort()
if len(image_files) == 0:
    raise RuntimeError("No FITS files found in folder_path")

star_coordinates = np.load(star_coordinates_loc)  # expected shape (N, >=4) with [?, ?, y, x]

# Process up to min(len(image_files), len(star_coordinates)); skip rows in star_coordinates that have NaN x/y
n_images = len(image_files)
n_coords = len(star_coordinates)
process_count = min(n_images, n_coords)

star_boxes = []
com_positions_in_boxes = []
hwhms = []
skipped_indices = []

for idx in range(process_count):
    raw_y = star_coordinates[idx][2]
    raw_x = star_coordinates[idx][3]
    if np.isnan(raw_x) or np.isnan(raw_y):
        skipped_indices.append(idx)
        continue  # skip this row entirely (no recovery)
    file_path = image_files[idx]
    data, hdr = open_fits(file_path)

    y_center = int(round(raw_y))
    x_center = int(round(raw_x))

    y_start = max(0, y_center - box_size // 2)
    y_end = min(data.shape[0], y_center + box_size // 2)
    x_start = max(0, x_center - box_size // 2)
    x_end = min(data.shape[1], x_center + box_size // 2)

    if y_end - y_start < 5 or x_end - x_start < 5:
        skipped_indices.append(idx)
        continue

    box = data[y_start:y_end, x_start:x_end]
    com_x, com_y = calculate_com(box)
    star_boxes.append(box)
    com_positions_in_boxes.append((com_x, com_y))

if len(star_boxes) == 0:
    raise RuntimeError("No valid star boxes collected (all frames skipped).")

max_radius = box_size // 2
radial_profiles = [calculate_radial_profile(box, com, max_radius) for box, com in zip(star_boxes, com_positions_in_boxes)]

for profile in radial_profiles:
    hwhm_pix = fit_fwhm_from_profile(profile)
    hwhms.append(hwhm_pix)

FWHM_pixels = np.array([2.0 * h for h in hwhms if not np.isnan(h)])
FWHM_arcsec = FWHM_pixels * pixel_scale

# print results
for idx, fwhm in enumerate(FWHM_pixels):
    print(f"Radial Profile {idx+1} FWHM: {fwhm:.3f} pixels  ({fwhm*pixel_scale:.3f} arcsec)")
print(f"Skipped coordinate rows (indices): {skipped_indices}")

# histogram (arcsec) with mean & median, save with same base name as .npy
median_fwhm = np.median(FWHM_arcsec) if FWHM_arcsec.size else np.nan
mean_fwhm = np.mean(FWHM_arcsec) if FWHM_arcsec.size else np.nan

fig_arc, ax_arc = plt.subplots(figsize=(8, 6))
ax_arc.hist(FWHM_arcsec, bins=10, edgecolor='k', alpha=0.6)
ax_arc.set_xlabel('FWHM (arcseconds)', fontsize=12)
ax_arc.set_ylabel('Density', fontsize=12)
ax_arc.set_title('Histogram of FWHM values (arcseconds)', fontsize=12)
ax_arc.text(0.95, 0.95, f'Median: {median_fwhm:.3f} arcsec\nMean: {mean_fwhm:.3f} arcsec',
            verticalalignment='top', horizontalalignment='right', transform=ax_arc.transAxes, fontsize=10,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
plt.tight_layout()

npy_base = os.path.splitext(os.path.basename(star_coordinates_loc))[0]
save_name = f"{npy_base}.png"
save_path = os.path.join(output_plots_dir, save_name)
fig_arc.savefig(save_path, dpi=300)
print(f"Saved FWHM (arcsec) histogram with mean/median to: {save_path}")

# optional: show star boxes grid for processed frames
num_profiles = len(radial_profiles)
grid_size = int(np.ceil(np.sqrt(num_profiles)))
fig_box, axes_box = plt.subplots(grid_size, grid_size, figsize=(15, 15))
axes_box = axes_box.flatten()
for idx, (ax, box) in enumerate(zip(axes_box, star_boxes)):
    norm = simple_norm(box, 'sqrt', percent=99)
    ax.imshow(box, origin='lower', cmap=color, norm=norm)
    ax.set_title(f'Box {idx+1}', fontsize=7, pad=3)
    ax.set_xticks([]); ax.set_yticks([])
for ax in axes_box[num_profiles:]:
    ax.axis('off')
plt.tight_layout()
plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.show()

print(f"Processed frames: {len(star_boxes)}  Skipped frames: {len(skipped_indices)}")
print(f"Median of PSF (arcsec): {median_fwhm:.3f}    Mean of PSF (arcsec): {mean_fwhm:.3f}")