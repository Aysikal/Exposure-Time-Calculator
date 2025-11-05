import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.visualization import ZScaleInterval, ImageNormalize

# === User config ===
fits_path = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\rezaei_saba_farideH_2025_10_22\GRB251013c\high\hot pixels removed\aligned\stacked\stacked-sum_RGB.fits"
grb_position = (2090.9, 1497.5)  # GRB approx position
cutout_half = 100
REFINE_BOX = 25

# Exposure and zero point
exptime_s = 6785  # total exposure time in seconds
ZP = 23.8768      # known zero point for this filter

# Instrument defaults
gain = 1/16.5
readnoise = 3.7

# SNR scan parameters
radius_step = 0.5
inner_radius_factor = 2.5
outer_radius_factor = 3.0

# Pixel scale and PSF constraints
pixel_scale_arcsec = 0.047 * 1.8   # arcsec / pixel
min_fwhm_arcsec = 1.0
min_fwhm_pixels = float(min_fwhm_arcsec / pixel_scale_arcsec)
min_hwhm_pixels = min_fwhm_pixels / 2.0

# --- Helper functions ---
def circular_mask(shape, center, radius):
    yy, xx = np.indices(shape)
    cx, cy = center
    return (xx - cx)**2 + (yy - cy)**2 <= radius**2

def annulus_mask(shape, center, r_in, r_out):
    yy, xx = np.indices(shape)
    cx, cy = center
    r2 = (xx - cx)**2 + (yy - cy)**2
    return (r2 >= r_in**2) & (r2 <= r_out**2)

def centroid_in_array(arr):
    y, x = np.indices(arr.shape)
    arr_pos = np.where(arr > 0, arr, 0.0)
    total = arr_pos.sum()
    if total <= 0:
        return arr.shape[1]/2.0, arr.shape[0]/2.0
    cx = (x * arr_pos).sum() / total
    cy = (y * arr_pos).sum() / total
    return cx, cy

def estimate_hwhm(cutout, center):
    img = cutout.astype(float)
    yy, xx = np.indices(img.shape)
    r = np.sqrt((xx - center[0])**2 + (yy - center[1])**2)
    r_flat = r.ravel()
    img_flat = img.ravel()
    r_max = max(4.0, min(img.shape)/2.0)
    nbins = int(np.ceil(r_max*4))
    bins = np.linspace(0.0, r_max, nbins+1)
    idx = np.digitize(r_flat, bins) - 1
    radial_med = np.array([np.median(img_flat[idx==i]) if np.any(idx==i) else 0.0 for i in range(len(bins)-1)])
    radii = 0.5*(bins[:-1] + bins[1:])
    peak = radial_med.max()
    if peak <= 0:
        return 1.0
    half = peak / 2.0
    try:
        j = np.where(radial_med <= half)[0][0]
        hwhm = radii[j]
        if hwhm <= 0:
            return 1.0
        return float(hwhm)
    except IndexError:
        return float(radii[-1])

def get_radius(image, center_xy, HWHM, gain, readnoise,
               radius_step=0.5,
               inner_radius=inner_radius_factor, outer_radius=outer_radius_factor):
    cx, cy = center_xy
    yy, xx = np.indices(image.shape)
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    image = image.astype(np.float64)

    HWHM = max(HWHM, min_hwhm_pixels)
    FWHM = max(2.0 * HWHM, min_fwhm_pixels)

    radius_min = max(1.0, 0.6 * FWHM)
    max_possible = max(1.0, min(image.shape) / 2.0 - 1.0)
    radius_max = min(max_possible, max(radius_min + radius_step, 3.5 * FWHM, 8.0))
    if radius_max <= radius_min:
        radius_max = radius_min + radius_step*4

    best_radius = radius_min
    max_snr = -np.inf
    snrs, radii_list = [], []

    for radius in np.arange(radius_min, radius_max + radius_step/2.0, radius_step):
        star_mask = dist <= radius
        n_star_pix = float(np.count_nonzero(star_mask))
        if n_star_pix <= 0:
            snrs.append(0.0)
            radii_list.append(radius)
            continue

        sum_brightness = float(np.nansum(image[star_mask]))
        ann_mask = (dist > inner_radius * radius) & (dist <= outer_radius * radius)
        ann_pixels = image[ann_mask]
        mean_bg = float(np.nanmedian(ann_pixels)) if ann_pixels.size>0 else float(np.nanmedian(image))
        net_counts = sum_brightness - n_star_pix * mean_bg

        S_e = net_counts * gain
        sky_e = n_star_pix * mean_bg * gain
        var_e = max(S_e, 0) + sky_e + n_star_pix*(readnoise**2)
        noise_e = np.sqrt(max(var_e, 1e-9))
        snr = S_e / noise_e if noise_e>0 else 0.0

        snrs.append(float(snr))
        radii_list.append(float(radius))
        if snr > max_snr:
            max_snr = float(snr)
            best_radius = float(radius)

    if max_snr == -np.inf:
        max_snr = 0.0
        best_radius = radius_min

    return best_radius, max_snr, snrs, radii_list

# === Load image ===
data = fits.getdata(fits_path)

# Preview
zscale = ZScaleInterval()
norm_full = ImageNormalize(data, interval=zscale)
plt.figure(figsize=(8,8))
plt.imshow(data, origin='lower', cmap='gray_r', norm=norm_full)
plt.plot(grb_position[0], grb_position[1], marker='+', color='yellow', ms=12)
plt.text(grb_position[0]+8, grb_position[1]+8, "GRB", color='yellow', fontsize=12)
plt.title(os.path.basename(fits_path))
plt.colorbar(label='ADU')
plt.tight_layout()
plt.show()

# --- Aperture photometry on GRB ---
tiny = Cutout2D(data, grb_position, (REFINE_BOX, REFINE_BOX), mode='partial')
cx_local, cy_local = centroid_in_array(tiny.data)
x_star, y_star = tiny.to_original_position((cx_local, cy_local))

disp_cut = Cutout2D(data, (x_star, y_star), (2*cutout_half, 2*cutout_half), mode='partial')
disp = np.nan_to_num(disp_cut.data)
cx_disp, cy_disp = disp_cut.to_cutout_position((x_star, y_star))

HWHM = estimate_hwhm(disp, (cx_disp, cy_disp))
HWHM_used = max(HWHM, min_hwhm_pixels)
FWHM_used = max(2.0*HWHM_used, min_fwhm_pixels)
print(f"Estimated HWHM = {HWHM_used:.2f}px (FWHM ~ {FWHM_used:.2f}px)")

best_r, max_snr, snrs, radii = get_radius(disp, (cx_disp, cy_disp), HWHM_used, gain, readnoise, radius_step)
print(f"Best radius = {best_r:.2f}px, max SNR = {max_snr:.2f}")

# Aperture photometry
ap_mask = circular_mask(disp.shape, (cx_disp, cy_disp), best_r)
ann_mask = annulus_mask(disp.shape, (cx_disp, cy_disp), inner_radius_factor*best_r, outer_radius_factor*best_r)
n_pix = np.count_nonzero(ap_mask)
sum_star = np.nansum(disp[ap_mask])
mean_bg = np.nanmedian(disp[ann_mask]) if np.any(ann_mask) else np.nanmedian(disp)
net_counts = sum_star - n_pix*mean_bg

net_per_sec = net_counts / exptime_s
GRB_mag = -2.5 * np.log10(net_per_sec) + ZP
instrumnetal_mag =  -2.5 * np.log10(net_per_sec)
print(f"Instrumental GRB magnitude = {instrumnetal_mag:.3f}")
print(f"\nCalibrated GRB magnitude = {GRB_mag:.3f}")
