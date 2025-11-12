import numpy as np
import os
from datetime import datetime
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
import logging
from astropy.stats import sigma_clipped_stats

# Constants
inner_radius_factor = 2.4
outer_radius_factor = 3
min_hwhm_pixels = 1
min_fwhm_pixels = 2
gain = 16.5
readnoise = 3.7  

# SNR-based aperture radius optimization
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

    radius_min = max(1.0, 0.6 * FWHM)
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
        sky_e = (n_star_pix * bg_level) * gain
        var_e = (sum_brightness/gain) + sky_e + n_star_pix * (readnoise ** 2)
        noise_e = np.sqrt(max(var_e, 1e-9))
        snr = S_e / noise_e if noise_e > 0 else 0.0

        snrs.append(snr)
        radii_list.append(radius)

        if snr > max_snr:
            max_snr = snr
            best_radius = radius

    return best_radius, max_snr, snrs, radii_list

# Input star and file
refined = [("05:58:25.03031399729", "+00:05:13.5242526788")]
file_to_test = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\Rezaei_Hossein_Atanaz_Kosar_2025_11_04\light\Aysan\high\hot pixels removed\97b-8\aligned\r\dark_corrected\aligned_97b_8_r_2025_11_05_1x1_exp00.00.01.000_000001_High_1_cycleclean_iter3_dark_corrected.fit"

# Load FITS and WCS
hdul = fits.open(file_to_test)
image_data = hdul[0].data
wcs = WCS(hdul[0].header)

# Convert RA/Dec to pixel coordinates
ra, dec = refined[0]
coord = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
center_xy = wcs.world_to_pixel(coord)

# Estimate HWHM and readnoise
HWHM = 5  # Adjust as needed

# Run the aperture optimization
best_radius, max_snr, snrs, radii_list = get_radius(
    image_data, center_xy, HWHM, gain, readnoise
)

# Output results
print("Best radius:", best_radius)
print("Max SNR:", max_snr)

# Plot SNR vs Radius
plt.plot(radii_list, snrs)
plt.xlabel("Radius (pixels)")
plt.ylabel("SNR")
plt.title("SNR vs Aperture Radius")
plt.grid(True)
plt.show()

