import numpy as np
import os
from datetime import datetime
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from ancillary_functions import airmass_function
import matplotlib.pyplot as plt
import logging

# -----------------------------
# USER PARAMETERS
# -----------------------------
e_to_ADU_gain = 1.0       # Physical gain: ADU → electrons
count_gain = 1/45         # Camera quirk: unitless scaling for SNR
READNOISE = 3.7
REFINE_BOX = 60
inner_radius_factor = 1.5
outer_radius_factor = 2.5
min_hwhm_pixels = 0.7
min_fwhm_pixels = 1.4

RA_HARD = "05:58:25.03"
DEC_HARD = "+00:06:40.7"

FILTER_FOLDER = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\Rezaei_Hossein_Atanaz_Kosar_2025_11_04\light\Aysan\high\hot pixels removed\97b-8\aligned\g\reduced"

logging.basicConfig(level=logging.INFO, format='%(message)s')

# -----------------------------
# Constants for efficiency calculation
# -----------------------------
h = 6.62607015e-34
c = 2.99792458e8
CW = {"u": 3560e-10, "g": 4825e-10, "r": 6261e-10, "i": 7672e-10}
bandwidth = {"u": 463e-10, "g": 988e-10, "r": 1340e-10, "i": 1064e-10}
extinction = {"u": 0.404, "g": 0.35, "r": 0.20, "i": 0.15}

D = 3.4
d = 0.6
S = np.pi * (D/2)**2 - np.pi * (d/2)**2  # collecting area

# -----------------------------
# Instrument throughput
# -----------------------------
R_mirror = 0.5      # reflectivity per mirror
n_mirrors = 2       # assume 2 mirrors
T_filter = 0.384    # filter transmittance
QE = 0.7            # detector quantum efficiency

throughput = (R_mirror**n_mirrors) * T_filter * QE

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
        mean_bg = float(np.nanmedian(ann_pixels)) if ann_pixels.size>0 else float(np.nanmedian(image))

        background_brightness = mean_bg * n_star_pix
        net_counts = sum_brightness - background_brightness

        S_e = net_counts * gain
        sky_e = n_star_pix * mean_bg * gain
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
# EXTRACT ADU
# -----------------------------
def extract_adu(image, x, y, r, inner_factor=inner_radius_factor, outer_factor=outer_radius_factor):
    yy, xx = np.indices(image.shape)
    dist = np.sqrt((xx - x)**2 + (yy - y)**2)

    star_mask = dist <= r
    n_pix = float(np.count_nonzero(star_mask))
    if n_pix == 0:
        return 0.0

    star_sum = float(np.nansum(image[star_mask]))
    ann_mask = (dist > inner_factor*r) & (dist <= outer_factor*r)
    ann_pixels = image[ann_mask]
    bg_level = float(np.nanmedian(ann_pixels)) if ann_pixels.size>0 else float(np.nanmedian(image))
    bg_total = bg_level * n_pix

    return float(star_sum - bg_total)

# -----------------------------
# EFFICIENCY FUNCTIONS
# -----------------------------
def mag_to_flux_lambda(mag, wav):
    f_nu = 10**(-0.4*(mag+48.6))
    f_lambda = f_nu * c / (wav**2)
    return f_lambda*1e-8

def expected_photons_per_second(mag, wav, bw, area, extinction_coeff, airmass):
    mag_atm = mag + extinction_coeff*airmass
    f_lambda_atm = mag_to_flux_lambda(mag_atm, wav)
    photon_energy = h*c/wav
    f_lambda_J = f_lambda_atm*1e-7
    bw_A = bw*1e10
    photons_m2_s = (f_lambda_J*bw_A)/photon_energy *1e4

    # APPLY TELESCOPE + FILTER + QE THROUGHOUT
    photons_m2_s *= throughput

    return photons_m2_s*area

def measured_photons_per_second(adu_sum, exposure_time, gain):
    return (adu_sum*gain)/exposure_time

def compute_efficiency(mag_catalog, adu_sum, exposure_time, filt, airmass):
    wav = CW[filt]
    bw = bandwidth[filt]
    ext = extinction[filt]
    expected = expected_photons_per_second(mag_catalog, wav, bw, S, ext, airmass)
    observed = measured_photons_per_second(adu_sum, exposure_time, e_to_ADU_gain)
    print(f"Expected photons/s: {expected:.2e}, Observed photons/s: {observed:.2e}")
    return observed/expected if expected>0 else 0.0

# -----------------------------
# MAIN FUNCTION FOR ONE FILE
# -----------------------------
def run_efficiency_one_file(file_path, refined_star_list_radec, filt, catalog_mag, grid_size=None):
    import math

    data_frame = fits.getdata(file_path)
    hdr = fits.getheader(file_path)
    wcs = WCS(hdr)

    exp = hdr.get("EXPTIME", np.nan)
    exp_real = exp*1e-5
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
        ncols = math.ceil(math.sqrt(n_stars))
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

        adu = extract_adu(data_frame, x_ref, y_ref, best_r)
        E = compute_efficiency(catalog_mag, adu, exp_real, filt, X)

        logging.info(f"{os.path.basename(file_path)} | Star {sid} | RA={ra_str}, DEC={dec_str}")
        logging.info(f"  Best radius: {best_r:.2f} px | SNR: {best_snr:.2f}")
        logging.info(f"  ADU: {adu:.2f} | Airmass: {X:.2f} | Efficiency: {E:.4f}")

        results_frame.append({
            "star_id": sid,
            "RA": ra_str,
            "DEC": dec_str,
            "r_best": best_r,
            "SNR": best_snr,
            "ADU": adu,
            "airmass": X,
            "exp_s": exp_real,
            "efficiency": E
        })

    for ax in axes_cutouts[n_stars:]:
        ax.axis('off')
    for ax in axes_snr[n_stars:]:
        ax.axis('off')

    plt.suptitle("Cutouts", fontsize=16)
    plt.tight_layout()
    plt.show()

    plt.suptitle("SNR vs Radius", fontsize=16)
    plt.tight_layout()
    plt.show()

    return results_frame

# -----------------------------
# MAIN CALL
# -----------------------------
refined = [("05:58:25.03031399729", "+00:05:13.5242526788")]
filt = "g"
catalog_mag = 11.455

file_to_test = os.path.join(FILTER_FOLDER, 
    "aligned_97b_8_g_2025_11_05_1x1_exp00.00.01.000_000001_High_2_cycleclean_iter3_dark_and_flat_corrected.fit")

results = run_efficiency_one_file(file_to_test, refined, filt, catalog_mag)
print(results)
