import numpy as np
import os
from datetime import datetime
from astropy.io import fits
from astropy.nddata import Cutout2D
import matplotlib.pyplot as plt
import logging
from astropy.stats import sigma_clipped_stats
from astropy.modeling import models, fitting

# ---------------- Constants ----------------
inner_radius_factor = 2.4
outer_radius_factor = 3
min_hwhm_pixels = 1
min_fwhm_pixels = 2
gain = 45
readnoise = 3.7

# ---------------- Function: SNR-based aperture radius optimization ----------------
def get_radius(image, center_xy, HWHM, gain, readnoise,
               radius_step=0.5,
               inner_radius=inner_radius_factor,
               outer_radius=outer_radius_factor):
    cx, cy = float(center_xy[0]), float(center_xy[1])
    yy, xx = np.indices(image.shape)
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    image = image.astype(np.float64)

    HWHM = max(HWHM if np.isfinite(HWHM) and HWHM > 0 else 1.0, min_hwhm_pixels)
    FWHM = max(2.0 * HWHM, min_fwhm_pixels)

    radius_min = 1.0
    max_possible = max(1.0, min(image.shape) / 2.0 - 1.0)
    radius_max = min(max_possible, max(radius_min + radius_step, 3.5 * FWHM, 12.0))
    if radius_max <= radius_min:
        radius_max = radius_min + radius_step * 4.0

    best_radius = radius_min
    max_snr = -np.inf
    snrs, radii_list = [], []

    for radius in np.arange(radius_min, radius_max + radius_step / 2.0, radius_step):
        star_mask = dist <= radius
        n_star_pix = float(np.count_nonzero(star_mask))
        if n_star_pix <= 0:
            snrs.append(0.0)
            radii_list.append(radius)
            continue

        sum_brightness = float(np.nansum(image[star_mask]))
        ann_mask = (dist > inner_radius * radius) & (dist <= outer_radius * radius)
        ann_pixels = image[ann_mask]
        if ann_pixels.size > 0:
            bg_level, _, _ = sigma_clipped_stats(ann_pixels, sigma=3.0)
            if not np.isfinite(bg_level):
                bg_level = float(np.nanmedian(image))
        else:
            bg_level = float(np.nanmedian(image))
        background_brightness = bg_level * n_star_pix
        net_counts = sum_brightness - background_brightness

        S_e = net_counts / gain
        sky_e = (n_star_pix * bg_level) / gain
        var_e = (sum_brightness / gain) + sky_e + n_star_pix * (readnoise ** 2)
        noise_e = np.sqrt(max(var_e, 1e-9))
        snr = S_e / noise_e if noise_e > 0 else 0.0

        snrs.append(snr)
        radii_list.append(radius)

        if snr > max_snr:
            max_snr = snr
            best_radius = radius

    return best_radius, max_snr, snrs, radii_list


# ---------------- User Inputs ----------------
# Direct pixel coordinates 
pixel_x, pixel_y = 780, 951  
center_xy = (pixel_x, pixel_y)

file_to_test = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\Sep30\Rezaei_30_sep_2025\target3\r\high\keep\hot pixels removed\aligned\reduced\aligned_target3_r_T10C_2025_10_01_2x2_exp00.01.00.000_000009_High_1_cycleclean_iter3_dark_and_flat_corrected.fit"

# ---------------- Load FITS ----------------
hdul = fits.open(file_to_test)
image_data = hdul[0].data

# ---------------- Estimate HWHM ----------------
HWHM = 5  # Adjust as needed

# ---------------- Initial SNR run ----------------
best_radius, max_snr, snrs, radii_list = get_radius(
    image_data, center_xy, HWHM, gain, readnoise
)

print("Initial center (x, y):", center_xy)
print("Initial best radius:", best_radius)
print("Initial Max SNR:", max_snr)

# ---------------- Plot initial SNR vs Radius ----------------
plt.figure()
plt.plot(radii_list, snrs, marker='o')
plt.xlabel("Radius (pixels)")
plt.ylabel("SNR")
plt.title("SNR vs Aperture Radius (Initial Center)")
plt.grid(True)
plt.show()

# ---------------- Refinement: fit 2D Gaussian on a local cutout ----------------
cutout_size = (50, 50)
initial_cutout = Cutout2D(image_data, position=center_xy, size=cutout_size)

data_cut = initial_cutout.data.copy()
data_for_fit = np.nan_to_num(data_cut, nan=0.0)
data_for_fit[data_for_fit < 0] = 0.0

ny, nx = data_for_fit.shape
y_grid, x_grid = np.mgrid[0:ny, 0:nx]

amp0 = np.nanmax(data_for_fit)
x0 = nx / 2.0
y0 = ny / 2.0
sigma_x0 = sigma_y0 = 3.0

gauss_init = models.Gaussian2D(amplitude=amp0,
                               x_mean=x0,
                               y_mean=y0,
                               x_stddev=sigma_x0,
                               y_stddev=sigma_y0,
                               theta=0.0)

fitter = fitting.LevMarLSQFitter()
try:
    with np.errstate(all='ignore'):
        gauss_fit = fitter(gauss_init, x_grid, y_grid, data_for_fit)
    x_fit = float(gauss_fit.x_mean.value)
    y_fit = float(gauss_fit.y_mean.value)
    if not (np.isfinite(x_fit) and np.isfinite(y_fit)):
        raise ValueError("Non-finite fit result")
    fit_success = True
except Exception as e:
    x_fit = x0
    y_fit = y0
    fit_success = False
    logging.warning("Gaussian fit failed, using cutout center. Error: %s", e)

# Convert fitted cutout coordinates to global coordinates
x_global = center_xy[0] - (nx / 2.0) + x_fit
y_global = center_xy[1] - (ny / 2.0) + y_fit
refined_center = (x_global, y_global)

print("Fit success:", fit_success)
print("Refined center (x, y):", refined_center)

# ---------------- Recalculate SNR using refined center ----------------
best_radius, max_snr, snrs, radii_list = get_radius(
    image_data, refined_center, HWHM, gain, readnoise
)

print("Refined best radius:", best_radius)
print("Refined Max SNR:", max_snr)

# ---------------- Plot refined SNR vs Radius ----------------
plt.figure()
plt.plot(radii_list, snrs, marker='o')
plt.xlabel("Radius (pixels)")
plt.ylabel("SNR")
plt.title("SNR vs Aperture Radius (Refined Center)")
plt.grid(True)
plt.show()

# ---------------- Create final cutout centered on refined center ----------------
final_cutout = Cutout2D(image_data, position=refined_center, size=cutout_size)
cut_data = final_cutout.data

# ---------------- Display refined cutout ----------------
plt.figure(figsize=(6, 6))
im = plt.imshow(cut_data, origin='lower', cmap='viridis', interpolation='none')
cbar = plt.colorbar(im)
cbar.set_label('Pixel Value')
plt.title("Refined Star Cutout (Centered on Refined Position)")

# Mark high-value pixels
high_val_mask = cut_data > 3500
ys, xs = np.where(high_val_mask)
plt.scatter(xs, ys, facecolors='none', edgecolors='red', s=80, label='> 3500')

# Mark refined centroid (center of cutout)
x_fit_centered = cut_data.shape[1] / 2.0
y_fit_centered = cut_data.shape[0] / 2.0
plt.plot(x_fit_centered, y_fit_centered, marker='+', color='white', markersize=12, label='Refined centroid')

# Overlay aperture at best_radius
circ_theta = np.linspace(0, 2 * np.pi, 200)
circ_x = x_fit_centered + best_radius * np.cos(circ_theta)
circ_y = y_fit_centered + best_radius * np.sin(circ_theta)
plt.plot(circ_x, circ_y, color='cyan', lw=1.2, label=f'Aperture r={best_radius:.2f}')

plt.legend()
plt.xlabel("X (pixels)")
plt.ylabel("Y (pixels)")
plt.grid(False)
plt.tight_layout()
plt.show()