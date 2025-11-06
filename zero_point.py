import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.nddata import Cutout2D
from scipy.optimize import curve_fit
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from astroplan import Observer
from zoneinfo import ZoneInfo
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# -------------------------
# Site configuration
# -------------------------
SITE_NAME = "INO"
SITE_LAT, SITE_LON, SITE_ELEV = 35.674, 51.3188, 3600
SITE_LOCATION = EarthLocation(lat=SITE_LAT*u.deg, lon=SITE_LON*u.deg, height=SITE_ELEV*u.m)
SITE_OBSERVER = Observer(location=SITE_LOCATION, timezone="UTC", name=SITE_NAME)

# -------------------------
# Observation time
# -------------------------
OBS_UTC = "2025-10-01 02:07:54"
obs_time = Time(OBS_UTC, format="iso", scale="utc")
obs_dt = obs_time.to_datetime(timezone=ZoneInfo("UTC"))

# -------------------------
# Airmass function
# -------------------------
def compute_airmass(RA, DEC):
    ra_h, ra_m, ra_s = map(float, RA.split(":"))
    ra_deg = ra_h*15 + ra_m*0.25 + ra_s*(0.25/60)
    d, m, s = map(float, DEC.split(":"))
    dec_sign = 1 if d >= 0 else -1
    dec_deg  = dec_sign*(abs(d) + m/60 + s/3600)
    coord = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame="icrs")
    alt = coord.transform_to(AltAz(obstime=obs_time, location=SITE_LOCATION)).alt.degree
    z_rad = np.radians(90 - alt)
    X = 1.0 / (np.cos(z_rad) + 0.50572*(6.07995 + np.degrees(z_rad))**(-1.6364))
    return X

# -------------------------
# User configuration
# -------------------------
fits_path = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\Sep30\Rezaei_30_sep_2025\target3\r\high\keep\hot pixels removed\aligned\reduced\aligned_target3_r_T10C_2025_10_01_2x2_exp00.01.00.000_000001_High_2_cycleclean_iter3_dark_and_flat_corrected.fit"
catalog_path = r"C:\Users\AYSAN\Desktop\project\INO\ETC\PAN-star\area95_stars_compact.csv"

RA_STR = "03:53:21"
DEC_STR = "-00:00:20"

star_IDs = [42, 30, 43, 74, 75, 76, 63, 48, 79, 65, 61, 28, 89, 2, 4, 70, 54, 18, 62, 82, 80, 13, 12]
star_positions_g = [
    (793, 893), (809, 1113), (807, 753), (1170, 509), (1159, 381), (1159, 328),
    (1583, 604), (1785, 581), (1409, 409), (1599, 409), (1466, 1063),
    (1710, 1033), (908, 1448), (1171, 1704), (1561, 1632), (589, 1423),
    (336, 1292), (1127, 1276), (1674, 704), (1963, 657), (1906, 880),
    (195, 1418)
]
star_positions_r = [
    (780, 950), (797, 1170), (794, 810), (1156, 565), (1145, 435), (1143, 382),
    (1568, 656), (1771, 633), (1394, 460), (1584, 460), (1452, 1117),
    (1696, 1085), (897, 1507), (1164, 1761), (1553, 1682), (580, 1483),
    (326, 1353), (1117, 1334), (1663, 757), (1951, 707), (1893, 932),
    (187, 1477), (163, 1530)
]

star_positions = star_positions_r

cutout_half = 50
REFINE_BOX = 50
gain = 1/16.5
readnoise = 3.7
inner_radius_factor = 2.5
outer_radius_factor = 3.0
k_r = 0.2  # extinction coefficient

# -------------------------
# Helper functions
# -------------------------
def circular_mask(shape, center, radius):
    yy, xx = np.indices(shape)
    cx, cy = center
    return (xx - cx)**2 + (yy - cy)**2 <= radius**2

def annulus_mask(shape, center, r_in, r_out):
    yy, xx = np.indices(shape)
    cx, cy = center
    r2 = (xx - cx)**2 + (yy - cy)**2
    return (r2 >= r_in**2) & (r2 <= r_out**2)

def gaussian_2d(xy, amp, x0, y0, sigma_x, sigma_y, theta, offset):
    x, y = xy
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    return offset + amp*np.exp(-(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2))

# -------------------------
# Load FITS and catalog
# -------------------------
data = fits.getdata(fits_path)
exptime_s = 58
catalog = pd.read_csv(catalog_path)

def get_known_mag(star_id):
    row = catalog[catalog['ID'] == star_id]
    if len(row) == 0: return np.nan
    return float(row['i_sdss'].values[0])

# -------------------------
# Compute airmass
# -------------------------
X = compute_airmass(RA_STR, DEC_STR)
print(f"Airmass at observation: X = {X:.3f}")

# -------------------------
# Photometry loop
# -------------------------
results = []
for star_id, (x_star, y_star) in zip(star_IDs, star_positions):
    known_mag = get_known_mag(star_id)
    if np.isnan(known_mag):
        continue

    try:
        tiny = Cutout2D(data, (x_star, y_star), (REFINE_BOX, REFINE_BOX), mode='partial')
    except:
        continue

    # Gaussian refinement
    y_idx, x_idx = np.indices(tiny.data.shape)
    x_flat, y_flat, data_flat = x_idx.ravel(), y_idx.ravel(), tiny.data.ravel()
    amp_guess = np.nanmax(tiny.data)
    x0_guess, y0_guess = REFINE_BOX/2, REFINE_BOX/2
    sigma_guess = 2.0
    theta_guess = 0.0
    offset_guess = np.nanmedian(tiny.data)
    p0 = [amp_guess, x0_guess, y0_guess, sigma_guess, sigma_guess, theta_guess, offset_guess]

    try:
        popt, _ = curve_fit(gaussian_2d, (x_flat, y_flat), data_flat, p0=p0, maxfev=3000)
        x0_fit, y0_fit = popt[1], popt[2]
        if 0 <= x0_fit <= REFINE_BOX and 0 <= y0_fit <= REFINE_BOX:
            x_refined = tiny.position_original[0] - REFINE_BOX/2 + x0_fit
            y_refined = tiny.position_original[1] - REFINE_BOX/2 + y0_fit
        else:
            x_refined, y_refined = tiny.position_original
    except:
        x_refined, y_refined = tiny.position_original

    # Aperture photometry
    try:
        disp_cut = Cutout2D(data, (x_refined, y_refined), (2*cutout_half, 2*cutout_half), mode='partial')
    except:
        continue
    disp = np.nan_to_num(disp_cut.data)
    cx, cy = cutout_half, cutout_half

    radii = np.arange(2, 13.1, 0.5)
    snrs = []
    for r in radii:
        ap_mask = circular_mask(disp.shape, (cx, cy), r)
        ann_mask = annulus_mask(disp.shape, (cx, cy), r*inner_radius_factor, r*outer_radius_factor)
        n_pix = np.count_nonzero(ap_mask)
        sum_star = np.nansum(disp[ap_mask])
        mean_bg = np.nanmedian(disp[ann_mask]) if np.any(ann_mask) else np.nanmedian(disp)
        net_counts = sum_star - n_pix*mean_bg
        S_e = net_counts*gain
        sky_e = n_pix*mean_bg*gain
        var_e = max(S_e,0)+sky_e+n_pix*(readnoise**2)
        noise_e = np.sqrt(max(var_e,1e-9))
        snr = S_e/noise_e if noise_e>0 else 0
        snrs.append(snr)

    best_r = radii[np.argmax(snrs)]
    max_snr = np.max(snrs)

    accepted = max_snr >= 3
    status_text = "ACCEPTED" if accepted else "REJECTED"

    if accepted:
        sum_star = np.nansum(disp[circular_mask(disp.shape, (cx, cy), best_r)])
        n_pix = np.count_nonzero(circular_mask(disp.shape, (cx, cy), best_r))
        mean_bg = np.nanmedian(disp[annulus_mask(disp.shape, (cx, cy), best_r*inner_radius_factor, best_r*outer_radius_factor)])
        net_counts = sum_star - n_pix*mean_bg
        net_per_sec = net_counts / exptime_s
        m_instr = -2.5*np.log10(net_per_sec) if net_per_sec>0 else np.nan
        ZP_raw = known_mag + 2.5*np.log10(net_per_sec) if net_per_sec>0 else np.nan
        ZP_corr = ZP_raw - k_r*(X-1) if net_per_sec>0 else np.nan

        results.append({
            'ID': star_id,
            'x_refined': x_refined,
            'y_refined': y_refined,
            'best_radius': best_r,
            'instr_mag': m_instr,
            'known_mag': known_mag,
            'ZP_raw': ZP_raw,
            'ZP_corr': ZP_corr
        })
    else:
        print(f"Skipping star ID {star_id} due to low SNR ({max_snr:.2f})")

    # -------------------------
    # Visualization
    # -------------------------
    plt.figure(figsize=(10,4))
    # SNR curve
    plt.subplot(1,2,1)
    plt.plot(radii, snrs, marker='o')
    plt.axvline(best_r, color='r', linestyle='--', label=f'Best radius={best_r:.1f}')
    plt.xlabel('Aperture radius [pix]')
    plt.ylabel('SNR')
    plt.title(f'Star ID {star_id} SNR Curve ({status_text})')
    plt.legend()

    # Cutout image
    plt.subplot(1,2,2)
    plt.imshow(disp, origin='lower', cmap='viridis')
    plt.colorbar(label='ADU')
    plt.title(f'Star ID {star_id} Cutout ({status_text})')
    circ_ap = Circle((cx, cy), best_r, edgecolor='red', facecolor='none', lw=2, label='Aperture')
    circ_in = Circle((cx, cy), best_r*inner_radius_factor, edgecolor='yellow', facecolor='none', lw=1, linestyle='--', label='Annulus inner')
    circ_out = Circle((cx, cy), best_r*outer_radius_factor, edgecolor='white', facecolor='none', lw=1, linestyle='--', label='Annulus outer')
    plt.gca().add_patch(circ_ap)
    plt.gca().add_patch(circ_in)
    plt.gca().add_patch(circ_out)
    plt.plot(cx, cy, marker='x', color='blue', markersize=10, label='Refined center')
    plt.legend()
    plt.tight_layout()
    plt.show()

# -------------------------
# Save results
# -------------------------
df = pd.DataFrame(results)
df.to_csv("photometry_results_i_extcorr.csv", index=False)
print("Saved results to photometry_results_i_extcorr.csv")

# -------------------------
# ZP statistics
# -------------------------
ZP_corr_values = df['ZP_corr'].dropna().values
mean_ZP = np.mean(ZP_corr_values)
median_ZP = np.median(ZP_corr_values)
std_ZP = np.std(ZP_corr_values, ddof=1)
error_ZP = std_ZP / np.sqrt(len(ZP_corr_values))

print(f"\n--- Extinction-Corrected ZP Statistics ---")
print(f"Mean ZP  = {mean_ZP:.4f}")
print(f"Median ZP= {median_ZP:.4f}")
print(f"Std Dev  = {std_ZP:.4f}")
print(f"Error    = {error_ZP:.4f}")
