import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
from astropy.time import Time
from astropy.coordinates import SkyCoord, AltAz, EarthLocation, get_body
import astropy.units as u

# ------------------------
# Observatory setup
# ------------------------
SITE_LAT = 33.674
SITE_LON = 51.3188
SITE_ELEV = 3600
SITE_TIMEZONE = "Asia/Tehran"
SITE_LOCATION = EarthLocation(lat=SITE_LAT*u.deg, lon=SITE_LON*u.deg, height=SITE_ELEV*u.m)

# If you need pixel scale later; not used in brightness calc
pixscale_arcsec = 0.101 * 2

# ------------------------
# Exponential fit parameters (mag vs Moon-target separation)
# ------------------------
a_fit, b_fit, c_fit = -3, 0.042, 19.705

def exp_decay(x, a, b, c):
    return a * np.exp(-b * x) + c

# ------------------------
# Moon illumination fraction
# ------------------------
def get_fli(date_str: str, hour: int, minute: int) -> float:
    tz_local = ZoneInfo(SITE_TIMEZONE)
    dt_local = datetime.strptime(f"{date_str} {hour:02d}:{minute:02d}", "%Y-%m-%d %H:%M").replace(tzinfo=tz_local)
    dt_utc   = dt_local.astimezone(ZoneInfo("UTC"))
    t_utc    = Time(dt_utc, scale="utc", location=SITE_LOCATION)

    moon = get_body("moon", t_utc, SITE_LOCATION)
    sun  = get_body("sun",  t_utc, SITE_LOCATION)

    psi = sun.separation(moon).to(u.rad).value  # radians
    fli = (1 - np.cos(psi)) / 2                 # 0..1
    return float(fli)

# ------------------------
# Sky magnitude calculation
# ------------------------
def calculate_sky_magnitude(date_str, hour, minute, ra, dec):
    # Target
    target_icrs = SkyCoord(ra=ra, dec=dec, unit=(u.hourangle, u.deg))

    # Time setup
    tz_local = ZoneInfo(SITE_TIMEZONE)
    dt_local = datetime.strptime(f"{date_str} {hour:02d}:{minute:02d}", "%Y-%m-%d %H:%M").replace(tzinfo=tz_local)
    dt_utc   = dt_local.astimezone(ZoneInfo("UTC"))
    t_utc    = Time(dt_utc, scale="utc", location=SITE_LOCATION)

    # Frames
    altaz_frame = AltAz(obstime=t_utc, location=SITE_LOCATION)
    target_altaz = target_icrs.transform_to(altaz_frame)
    alt_deg = float(target_altaz.alt.degree)

    # Airmass (simple sec z, with guards)
    if alt_deg <= 0:
        return np.inf  # target below horizon → undefined, return very bright (or handle upstream)
    z_deg = 90.0 - alt_deg
    airmass = 1.0 / np.cos(np.radians(z_deg))
    # Clamp absurd values near horizon
    airmass = np.clip(airmass, 1.0, 10.0)

    # Moon-target separation
    moon = get_body("moon", t_utc, SITE_LOCATION)
    moon_altaz = moon.transform_to(altaz_frame)
    moon_sep = float(target_altaz.separation(moon_altaz).degree)

    # Base sky mag from separation
    sky_mag_fit = exp_decay(moon_sep, a_fit, b_fit, c_fit)

    # Extinction correction
    EXTINCTION_COEFF = 0.2
    sky_mag_ext_corr = sky_mag_fit - EXTINCTION_COEFF * airmass

    # Fraction of Moon illuminated (ensure strictly > 0 to avoid log10(0))
    fli_raw = get_fli(date_str, hour, minute)
    fli = max(float(fli_raw), 1e-3)

    # Phase correction only if Moon is above horizon
    if float(moon_altaz.alt.deg) > 0:
        phase_correction = -2.5 * np.log10(fli)
    else:
        phase_correction = 0.0

    # Final sky magnitude
    sky_mag_final = sky_mag_ext_corr + phase_correction
    return float(sky_mag_final)

# ------------------------
# Example usage
# ------------------------
if __name__ == "__main__":
    date_str = "2025-10-7"
    hour = 22
    minute = 11
    ra = "03:53:21"
    dec = "-00:00:20"

    sky_mag = calculate_sky_magnitude(date_str, hour, minute, ra, dec)
    print(f"\nPredicted sky magnitude (mag/arcsec²): {sky_mag:.2f}")

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from astroplan import Observer
from astropy import units as u
from astropy.wcs import WCS
from scipy.optimize import curve_fit
from zoneinfo import ZoneInfo
import matplotlib.pyplot as plt

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
OBS_UTC = "2025-09-30 22:11:54"
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
star_positions_r = [
    (780, 950), (797, 1170), (794, 810), (1156, 565), (1145, 435), (1143, 382),
    (1568, 656), (1771, 633), (1394, 460), (1584, 460), (1452, 1117),
    (1696, 1085), (897, 1507), (1164, 1761), (1553, 1682), (580, 1483),
    (326, 1353), (1117, 1334), (1663, 757), (1951, 707), (1893, 932),
    (187, 1477), (163, 1530)
]
star_positions = star_positions_r

cutout_half = 50
inner_radius_factor = 2.5
outer_radius_factor = 3.0
k_r = 0.2  # extinction coefficient

# -------------------------
# Helper functions
# -------------------------
def gaussian_2d(xy, amp, x0, y0, sigma_x, sigma_y, theta, offset):
    x, y = xy
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    return offset + amp*np.exp(-(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2))

def aperture_sum(image, x0, y0, r_ap, r_in, r_out):
    yy, xx = np.indices(image.shape)
    rr2 = (xx - x0)**2 + (yy - y0)**2
    aper_mask = rr2 <= r_ap**2
    ann_mask  = (rr2 >= r_in**2) & (rr2 <= r_out**2)
    ann_vals = image[ann_mask]
    sky_mean, sky_med, sky_std = sigma_clipped_stats(ann_vals, sigma=3.0)
    F_tot = image[aper_mask].sum()
    N_aper = aper_mask.sum()
    F_net = F_tot - sky_mean * N_aper
    return F_net, sky_mean, N_aper

def gaussian_fwhm(sx, sy):
    return 2.355 * np.array([sx, sy])

def estimate_fwhm(image, x0, y0, half=20):
    x0, y0 = int(x0), int(y0)
    x1, x2 = max(0, x0-half), min(image.shape[1], x0+half)
    y1, y2 = max(0, y0-half), min(image.shape[0], y0+half)
    cut = image[y1:y2, x1:x2]
    yy, xx = np.indices(cut.shape)
    amp0 = cut.max() - cut.min()
    xg0, yg0 = (xx*cut).sum()/cut.sum(), (yy*cut).sum()/cut.sum()
    p0 = [amp0, xg0, yg0, 2.0, 2.0, 0.0, cut.min()]
    try:
        popt, _ = curve_fit(gaussian_2d, (xx, yy), cut.ravel(), p0=p0, maxfev=5000)
        fwhm_x, fwhm_y = gaussian_fwhm(popt[3], popt[4])
        fwhm_pix = float(np.mean([fwhm_x, fwhm_y]))
    except Exception:
        fwhm_pix = 3.0
    return fwhm_pix

# -------------------------
# Load FITS and catalog
# -------------------------
hdu = fits.open(fits_path)
data = hdu[0].data.astype(np.float64)
hdr = hdu[0].header
hdu.close()

exptime_s = 58
catalog = pd.read_csv(catalog_path)

def get_known_mag(star_id):
    row = catalog[catalog['ID'] == star_id]
    if len(row) == 0: return np.nan
    return float(row['i_sdss'].values[0])  # adjust to r_sdss if available

# -------------------------
# Compute airmass
# -------------------------
X = compute_airmass(RA_STR, DEC_STR)
print(f"Airmass at observation: X = {X:.3f}")

# -------------------------
# Zero point calibration
# -------------------------
ZPs = []
for sid, (x, y) in zip(star_IDs, star_positions):
    m_cat = get_known_mag(sid)
    if not np.isfinite(m_cat):
        continue
    fwhm = estimate_fwhm(data, x, y, half=cutout_half)
    r_ap  = inner_radius_factor * fwhm
    r_in  = outer_radius_factor * fwhm
    r_out = (outer_radius_factor + 1.0) * fwhm
    F_net, sky_mean, N_aper = aperture_sum(data, x, y, r_ap, r_in, r_out)
    if F_net <= 0:
        continue
    m_inst = -2.5 * np.log10(F_net / exptime_s)
    m_corr = m_inst + k_r * X
    ZP = m_cat - m_corr
    ZPs.append(ZP)

ZP_final = np.median(ZPs)
print(f"Photometric zero point (median): {ZP_final:.3f} mag")

# ---- Select an empty sky patch and integrate ----
# Define a box center and half-size (adjust to a truly empty region)
sky_xc, sky_yc = 1200, 900
sky_half = 50  # pixels
x1, x2 = max(0, sky_xc - sky_half), min(data.shape[1], sky_xc + sky_half)
y1, y2 = max(0, sky_yc - sky_half), min(data.shape[0], sky_yc + sky_half)
sky_patch = data[y1:y2, x1:x2]

# Sigma-clipped statistics of the patch
sky_mean, sky_med, sky_std = sigma_clipped_stats(sky_patch, sigma=3.0)
A_pix = sky_patch.size

# Integrated flux in the patch
F_sky_total = sky_patch.sum()

# Guard against zero flux
if F_sky_total <= 0:
    raise ValueError("Sky patch flux is non-positive; choose a different region.")

# Convert to instrumental magnitude (integrated patch)
m_sky_inst = -2.5 * np.log10(F_sky_total / exptime_s)

# Apply extinction and zero point to get calibrated sky magnitude for the patch
m_sky = m_sky_inst + k_r * X + ZP_final

# Convert to mag per arcsec^2
A_arcsec2 = A_pix * (pixscale_arcsec**2)
mu_sky = m_sky + 2.5 * np.log10(A_arcsec2)

print(f"Sky magnitude (integrated patch): {m_sky:.3f} mag")
print(f"Sky surface brightness: {mu_sky:.3f} mag/arcsec^2")
print(f"Patch area: {A_pix} px = {A_arcsec2:.2f} arcsec^2; plate scale: {pixscale_arcsec:.3f} arcsec/px")

