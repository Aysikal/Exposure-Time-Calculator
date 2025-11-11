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
from ancillary_functions import airmass_function

# -----------------------------
# USER PARAMETERS
# -----------------------------
CCD_GAIN = 1         # electrons per ADU (physical gain)
count_gain = 1.0/16.5     # Camera quirk: scaling for SNR (keep for get_radius)
READNOISE = 3.7          # electrons
REFINE_BOX = 60
inner_radius_factor = 1.5
outer_radius_factor = 2.5
min_hwhm_pixels = 0.7
min_fwhm_pixels = 1.4

RA_HARD = "05:58:25.03"
DEC_HARD = "+00:06:40.7"

FILTER_FOLDER = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\Rezaei_Hossein_Atanaz_Kosar_2025_11_04\light\Aysan\high\hot pixels removed\97b-8\aligned\r\reduced"

logging.basicConfig(level=logging.INFO, format='%(message)s')

# -----------------------------
# Constants for flux calculation
# -----------------------------
h = 6.62607015e-34
c = 2.99792458e8
CW = {"u": 3560e-10, "g": 4825e-10, "r": 6261e-10, "i": 7672e-10}
bandwidth = {"u": 463e-10, "g": 988e-10, "r": 1340e-10, "i": 1064e-10}
extinction = {"u": 0.404, "g": 0.35, "r": 0.20, "i": 0.15}

D = 3.4  # m1 
d = 0.6  # m2
S = np.pi * (D/2)**2 - np.pi * (d/2)**2  # collecting area in m^2
S_cm2 = S * 1e4
# -----------------------------
# SNR-OPTIMIZED RADIUS
# -----------------------------
def get_radius(image, center_xy, HWHM, gain, readnoise,
               radius_step=0.5,
               inner_radius=inner_radius_factor,
               outer_radius=outer_radius_factor):
    cx, cy = float(center_xy[0]), float(center_xy[1])
    yy, xx = np.indices(image.shape)
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    image = image.astype(np.float64)

    HWHM = max(HWHM if np.isfinite(HWHM) and HWHM>0 else 1.0, min_hwhm_pixels)
    FWHM = max(2.0*HWHM, min_fwhm_pixels)

    radius_min = max(1.0, 0.6*FWHM)
    max_possible = max(1.0, min(image.shape)/2.0 - 1.0)
    radius_max = min(max_possible, max(radius_min + radius_step, 3.5*FWHM, 12.0))
    if radius_max <= radius_min:
        radius_max = radius_min + radius_step*4.0

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
        ann_mask = (dist > inner_radius*radius) & (dist <= outer_radius*radius)
        ann_pixels = image[ann_mask]
        if ann_pixels.size>0:
            bg_level, _, _ = sigma_clipped_stats(ann_pixels, sigma=3.0)
        else:
            bg_level = float(np.nanmedian(image))
        background_brightness = bg_level * n_star_pix
        net_counts = sum_brightness - background_brightness

        S_e = net_counts * gain
        sky_e = n_star_pix * bg_level * gain
        var_e = max(S_e,0.0) + sky_e + n_star_pix*(readnoise**2)
        noise_e = np.sqrt(max(var_e,1e-9))
        snr = S_e / noise_e if noise_e>0 else 0.0

        snrs.append(snr)
        radii_list.append(radius)

        if snr > max_snr:
            max_snr = snr
            best_radius = radius

    return best_radius, max_snr, snrs, radii_list

# -----------------------------
# ADU Extraction
# -----------------------------
def extract_adu(image, x, y, r, inner_factor=inner_radius_factor, outer_factor=outer_radius_factor):
    yy, xx = np.indices(image.shape)
    dist = np.sqrt((xx - x)**2 + (yy - y)**2)
    star_mask = dist <= r
    n_pix = float(np.count_nonzero(star_mask))
    if n_pix == 0:
        return 0.0, n_pix

    star_sum = float(np.nansum(image[star_mask]))
    ann_mask = (dist > inner_factor*r) & (dist <= outer_factor*r)
    ann_pixels = image[ann_mask]
    if ann_pixels.size>0:
        bg_level, _, _ = sigma_clipped_stats(ann_pixels, sigma=3.0)
    else:
        bg_level = float(np.nanmedian(image))
    bg_total = bg_level * n_pix

    return (float(star_sum - bg_total))/45, n_pix

# -----------------------------
# Efficiency Functions (photons/sec)
# -----------------------------
def mag_to_flux_lambda(mag, wav):
    f_nu = 10**(-0.4*(mag+48.6))  # AB system
    f_lambda = (f_nu * c) / (wav ** 2) * 1e-10
    return f_lambda   

def expected_photons_per_second(mag, wav, bw, area_cm2, extinction_coeff, airmass):
    mag_atm = mag + extinction_coeff*airmass
    f_lambda_atm = mag_to_flux_lambda(mag_atm, wav)  # erg/cm^2/s/Å
    photon_energy = h*c/wav  # Joules
    f_lambda_J = f_lambda_atm * 1e-7  # erg → Joules
    bw_A = bw * 1e10  # meters → Å
    photons_per_s_per_cm2 = f_lambda_J * bw_A / photon_energy
    return photons_per_s_per_cm2 * area_cm2

def measured_photons_per_second(adu_sum, exposure_time, gain):
    # Convert ADU to e
    return (adu_sum * gain) / exposure_time

def compute_efficiency(mag_catalog, adu_sum, exposure_time, filt, airmass, n_pix):
    wav = CW[filt]
    bw = bandwidth[filt]
    ext = extinction[filt]

    expected = expected_photons_per_second(mag_catalog, wav, bw, S_cm2, ext, airmass)
    observed = measured_photons_per_second(adu_sum, exposure_time, CCD_GAIN)

    eff_percent = 100 * observed / expected if expected>0 else 0.0
    print(f"Expected photons/s: {expected:.2e}, Observed electrons/s: {observed:.2e}, Efficiency: {eff_percent:.2f}%")
    return eff_percent

# -----------------------------
# Main function for one FITS file
# -----------------------------
def run_efficiency_one_file(file_path, refined_star_list_radec, filt, catalog_mag, grid_size=None):
    import math

    data_frame = fits.getdata(file_path)
    hdr = fits.getheader(file_path)
    wcs = WCS(hdr)

    exp = hdr.get("EXPTIME", np.nan)
    exp_real = exp * 1e-5  # already consistent with your FITS
    print("EXPTIME =", exp_real)

    date_str_full = hdr.get("DATE")
    try:
        dt_obj = datetime.fromisoformat(date_str_full)
    except ValueError:
        dt_obj = datetime.strptime(date_str_full, "%Y-%m-%dT%H:%M:%S.%f")
    hour, minute = dt_obj.hour, dt_obj.minute
    X = airmass_function(dt_obj.strftime("%Y-%m-%d"), hour, minute, RA_HARD, DEC_HARD)
    print("Airmass =", X)

    n_stars = len(refined_star_list_radec)
    if grid_size is None:
        ncols = math.ceil(np.sqrt(n_stars))
        nrows = math.ceil(n_stars/ncols)
    else:
        nrows, ncols = grid_size

    fig_cutouts, axes_cutouts = plt.subplots(nrows, ncols, figsize=(4*ncols,4*nrows))
    axes_cutouts = np.array(axes_cutouts).reshape(-1)
    fig_snr, axes_snr = plt.subplots(nrows, ncols, figsize=(4*ncols,4*nrows))
    axes_snr = np.array(axes_snr).reshape(-1)

    results_frame = []

    for sid, (ra_str, dec_str) in enumerate(refined_star_list_radec):
        coord = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))
        x_ref, y_ref = wcs.world_to_pixel(coord)

        cutout = Cutout2D(data_frame, (x_ref, y_ref), REFINE_BOX)
        ax_c = axes_cutouts[sid]
        im = ax_c.imshow(cutout.data, origin='lower', cmap='gray')
        ax_c.set_title(f"Star {sid}")
        ax_c.axis('off')
        fig_cutouts.colorbar(im, ax=ax_c, fraction=0.046, pad=0.04)

        best_r, best_snr, s_list, r_list = get_radius(
            data_frame, (x_ref, y_ref), HWHM=1.0,
            gain=count_gain, readnoise=READNOISE
        )

        ax_s = axes_snr[sid]
        ax_s.plot(r_list, s_list, marker='o')
        ax_s.set_title(f"Star {sid}")
        ax_s.set_xlabel("Radius (px)")
        ax_s.set_ylabel("SNR")
        ax_s.grid(True)

        adu, n_pix = extract_adu(data_frame, x_ref, y_ref, best_r)
        E_percent = compute_efficiency(catalog_mag, adu, exp_real, filt, X, n_pix)

        logging.info(f"{os.path.basename(file_path)} | Star {sid} | RA={ra_str}, DEC={dec_str}")
        logging.info(f"  Best radius: {best_r:.2f} px | SNR: {best_snr:.2f}")
        logging.info(f"  ADU: {adu:.2f} | Airmass: {X:.2f} | Efficiency: {E_percent:.2f}%")

        results_frame.append({
            "star_id": sid,
            "RA": ra_str,
            "DEC": dec_str,
            "r_best": best_r,
            "SNR": best_snr,
            "ADU": adu,
            "n_pix": n_pix,
            "airmass": X,
            "exp_s": exp_real,
            "efficiency_percent": E_percent
        })

    for ax in axes_cutouts[n_stars:]:
        ax.axis('off')
    for ax in axes_snr[n_stars:]:
        ax.axis('off')

    plt.suptitle("Cutouts", fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()

    plt.suptitle("SNR vs Radius", fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()

    return results_frame

# -----------------------------
# MAIN CALL
# -----------------------------
refined = [("05:58:25.03031399729", "+00:05:13.5242526788")]
filt = "r"
catalog_mag = 10.1773171

file_to_test = os.path.join(FILTER_FOLDER,
    r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\Rezaei_Hossein_Atanaz_Kosar_2025_11_04\light\Aysan\high\hot pixels removed\97b-8\aligned\r\dark_corrected\aligned_97b_8_r_2025_11_05_1x1_exp00.00.01.000_000001_High_1_cycleclean_iter3_dark_corrected.fit"
)

results = run_efficiency_one_file(file_to_test, refined, filt, catalog_mag)
print(results)
