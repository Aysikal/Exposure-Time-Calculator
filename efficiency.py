# Corrected full Script B matching SNR logic of Script A
from zoneinfo import ZoneInfo
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
from ancillary_functions import airmass_function, calculate_sky_magnitude

# -----------------------------
# USER PARAMETERS (UPDATED TO MATCH SCRIPT A)
# -----------------------------
count_gain = 45
READNOISE = 3.7
REFINE_BOX = 60

# EXACT PARAMETERS FROM SCRIPT A
inner_radius_factor = 2.4
outer_radius_factor = 3.0
min_hwhm_pixels = 1.0
min_fwhm_pixels = 2.0
HWHM_FOR_SNR = 5.0

RA_HARD = "03:53:21"
DEC_HARD = "+00:00:20"

FILTER_FOLDER = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\Sep30\Rezaei_30_sep_2025\target3\r\high\keep\hot pixels removed\aligned\reduced\aligned_target3_r_T10C_2025_10_01_2x2_exp00.01.00.000_000009_High_1_cycleclean_iter3_dark_and_flat_corrected.fit"

logging.basicConfig(level=logging.INFO, format='%(message)s')

# -----------------------------
# Constants for flux calculations
# -----------------------------
h = 6.62607015e-34
c = 2.99792458e8

CW = {'u': 3540e-10, 'g': 4770e-10, 'r': 6230e-10, 'i': 7630e-10}
bandwidth = {'u': 600e-10, 'g': 1380e-10, 'r': 1380e-10, 'i': 1520e-10}
extinction = {"u": 0.404, "g": 0.35, "r": 0.20, "i": 0.15}

D = 3.4
d = 0.6
S = np.pi * (D / 2.0) ** 2 - np.pi * (d / 2.0) ** 2
S_cm2 = S * 1e4

# -----------------------------
# Corrected get_radius identical to Script A
# -----------------------------
def get_radius(image, center_xy, HWHM, gain, readnoise,
               radius_step=0.5,
               inner_radius=inner_radius_factor,
               outer_radius=outer_radius_factor):

    cx, cy = float(center_xy[0]), float(center_xy[1])
    yy, xx = np.indices(image.shape)
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    image = image.astype(np.float64)

    HWHM = max(HWHM if np.isfinite(HWHM) and HWHM > 0 else 1.0,
               min_hwhm_pixels)
    FWHM = max(2.0 * HWHM, min_fwhm_pixels)

    radius_min = max(1.0, 0.6 * FWHM)
    max_possible = max(1.0, min(image.shape) / 2.0 - 1.0)
    radius_max = min(max_possible,
                     max(radius_min + radius_step, 3.5 * FWHM, 12.0))

    if radius_max <= radius_min:
        radius_max = radius_min + radius_step * 4.0

    best_radius = radius_min
    max_snr = -np.inf
    best_S_e = 0.0
    best_npix = 0
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
            best_S_e = S_e
            best_npix = int(n_star_pix)

    return best_radius, max_snr, best_S_e, best_npix, snrs, radii_list

def expected_photons_per_second_v2(mag_ab, wav_m, bandwidth_m, area_cm2,
                                   extinction_coeff, airmass, sky_mag,
                                   npix=1, pixel_scale=1.0, readnoise=0.0, efficiency=1.0):
    """
    Compute expected photons per second from a star and sky,
    scaled to the aperture size and including readnoise.

    Parameters
    ----------
    mag_ab : float
        Star magnitude (AB system)
    wav_m : float
        Central wavelength (meters)
    bandwidth_m : float
        Filter bandwidth (meters)
    area_cm2 : float
        Telescope collecting area (cm^2)
    extinction_coeff : float
        Atmospheric extinction coefficient
    airmass : float
        Observing airmass
    sky_mag : float
        Sky background magnitude (AB system)
    npix : int
        Number of pixels in the aperture
    pixel_scale : float
        Pixel scale in arcseconds/pixel (or matching units)
    readnoise : float
        Read noise per pixel (electrons)
    efficiency : float
        System efficiency (0-1)
    
    Returns
    -------
    photons_star : float
        Expected star photons per second
    N_sky_e_per_pix : float
        Expected sky electrons per pixel per second
    """
    # --- Star flux corrected for extinction ---
    m_atm = mag_ab + extinction_coeff * airmass
    f_nu_star_erg = 10**(-0.4 * (m_atm + 48.6))
    f_nu_star_J = f_nu_star_erg * 1e-7
    f_lambda_star_J_per_m = f_nu_star_J * c / (wav_m ** 2)
    power_star_per_cm2 = f_lambda_star_J_per_m * bandwidth_m
    E_photon = h * c / wav_m
    photons_star = power_star_per_cm2 / E_photon * area_cm2
    electrons_star = photons_star * efficiency

    # --- Sky background scaled to aperture ---
    f_nu_s_erg = 10**(-0.4 * (sky_mag + 48.6))
    f_nu_s_J = f_nu_s_erg * 1e-7
    f_lambda_s_J_per_m = f_nu_s_J * c / (wav_m ** 2)
    power_sky_per_cm2 = f_lambda_s_J_per_m * bandwidth_m
    total_sky_power_J_per_s = power_sky_per_cm2 * area_cm2
    photons_sky_per_s = total_sky_power_J_per_s / E_photon
    electrons_sky_per_s = photons_sky_per_s * efficiency

    # Scale sky to aperture (per pixel)
    C_e_per_sec_per_pix = electrons_sky_per_s * (pixel_scale ** 2)
    N_sky_e_per_pix = C_e_per_sec_per_pix + readnoise**2  # includes readnoise contribution

    return electrons_star - (N_sky_e_per_pix*npix)


def measured_electrons_per_second(adu_sum, exposure_time):
    if exposure_time <= 0:
        return 0.0
    return (adu_sum) / exposure_time

# -----------------------------
# Main routine
# -----------------------------
def run_efficiency_one_file(file_path, refined_star_list_radec, filt, catalog_mag, grid_size=None):
    import math

    data_frame = fits.getdata(file_path)
    hdr = fits.getheader(file_path)
    wcs = WCS(hdr)

    exp = hdr.get("EXPTIME", np.nan)
    exp_real = exp * 1e-5
    print("EXPTIME =", exp_real)

    date_str_full = hdr.get("DATE")
    try:
        dt_obj = datetime.fromisoformat(date_str_full)
    except ValueError:
        dt_obj = datetime.strptime(date_str_full, "%Y-%m-%dT%H:%M:%S.%f")

    print("UTC datetime:", dt_obj.strftime("%Y-%m-%d %H:%M:%S"))

    hour, minute = dt_obj.hour, dt_obj.minute
    X = airmass_function(dt_obj.strftime("%Y-%m-%d"), hour, minute, RA_HARD, DEC_HARD)
    print("Airmass =", X)

    n_stars = len(refined_star_list_radec)
    if grid_size is None:
        ncols = math.ceil(np.sqrt(n_stars))
        nrows = math.ceil(n_stars / ncols)
    else:
        nrows, ncols = grid_size

    fig_cutouts, axes_cutouts = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes_cutouts = np.array(axes_cutouts).reshape(-1)
    fig_snr, axes_snr = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
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

        best_r, best_snr, best_S_e, best_npix, s_list, r_list = get_radius(
            data_frame,
            (x_ref, y_ref),
            HWHM=HWHM_FOR_SNR,
            gain=count_gain,
            readnoise=READNOISE
        )

        ax_s = axes_snr[sid]
        ax_s.plot(r_list, s_list, marker='o')
        ax_s.set_title(f"Star {sid}")
        ax_s.set_xlabel("Radius (px)")
        ax_s.set_ylabel("SNR")
        ax_s.grid(True)

        adu = best_S_e
        wav_m = CW[filt]
        bw_m = bandwidth[filt]
        ext_coeff = extinction[filt]
        # Convert dt_obj to local string in the expected format
        date_str_only = dt_obj.strftime("%Y-%m-%d")  # just the date
        hour = dt_obj.hour
        minute = dt_obj.minute

        sky_mag = calculate_sky_magnitude(date_str_only, hour, minute, RA_HARD, DEC_HARD)
       
        expected_ph_s = expected_photons_per_second_v2(catalog_mag, wav_m, bw_m, S_cm2, ext_coeff, X, sky_mag)

        observed_e_s = measured_electrons_per_second(adu, exp_real)
        E_percent = 100.0 * observed_e_s / expected_ph_s if expected_ph_s > 0 else 0.0

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
            "n_pix": best_npix,
            "airmass": X,
            "exp_s": exp_real,
            "efficiency_percent": E_percent
        })

    for ax in axes_cutouts[n_stars:]:
        ax.axis('off')
    for ax in axes_snr[n_stars:]:
        ax.axis('off')

    plt.suptitle("Cutouts", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    return results_frame

# -----------------------------
# MAIN CALL
# -----------------------------
refined = [("05:58:25.74", "+00:07:17.84")]
filt = "r"
catalog_mag = 14.1644001


file_to_test = os.path.join(FILTER_FOLDER,
    r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\Rezaei_Hossein_Atanaz_Kosar_2025_11_04\light\Aysan\high\hot pixels removed\97b-8\aligned\r\dark_corrected\aligned_97b_8_r_2025_11_05_1x1_exp00.00.01.000_000001_High_1_cycleclean_iter3_dark_corrected.fit"
)

results = run_efficiency_one_file(file_to_test, refined, filt, catalog_mag)
print(results)
