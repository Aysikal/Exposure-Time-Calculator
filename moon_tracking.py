# ───────────────────────────────────────────────────────────────
# Full Moon-Sky Background Analysis with 2D Sky Model
# Author: Aysan Hemmatiortakand
# Last updated: 10/07/2025
# ───────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, AltAz, EarthLocation, get_body
import astropy.units as u
from scipy.optimize import curve_fit
from datetime import datetime

# ───────────────────────────────────────────────────────────────
# Observatory setup
# ───────────────────────────────────────────────────────────────
SITE_NAME     = "Iran National Observatory"
SITE_LAT      = 33.674
SITE_LON      = 51.3188
SITE_ELEV     = 3600
SITE_LOCATION = EarthLocation(lat=SITE_LAT*u.deg, lon=SITE_LON*u.deg, height=SITE_ELEV*u.m)

# ───────────────────────────────────────────────────────────────
# Helper: convert Timedelta or Excel time to 'H:M:S' string
# ───────────────────────────────────────────────────────────────
def timedelta_to_hms(td):
    if isinstance(td, pd.Timedelta):
        total_seconds = td.total_seconds()
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60
        return f"{hours}:{minutes:02d}:{seconds:05.2f}"
    return str(td)

# ───────────────────────────────────────────────────────────────
# Function: Moon-target angular separation
# ───────────────────────────────────────────────────────────────
def moon_separation(t_utc, ra, dec):
    target_icrs = SkyCoord(ra=ra, dec=dec, unit=(u.hourangle, u.deg))
    altaz_frame = AltAz(obstime=t_utc, location=SITE_LOCATION)
    target_altaz = target_icrs.transform_to(altaz_frame)
    moon_altaz = get_body("moon", t_utc, SITE_LOCATION).transform_to(altaz_frame)
    return target_altaz.separation(moon_altaz).degree, target_altaz.alt.degree

# ───────────────────────────────────────────────────────────────
# Function: Moon illuminated fraction
# ───────────────────────────────────────────────────────────────
def get_fli(t_utc):
    moon = get_body("moon", t_utc, SITE_LOCATION)
    sun  = get_body("sun", t_utc, SITE_LOCATION)

    # Moon alt/az
    altaz_frame = AltAz(obstime=t_utc, location=SITE_LOCATION)
    moon_altaz = moon.transform_to(altaz_frame)
    alt, az = moon_altaz.alt.deg, moon_altaz.az.deg

    # Geocentric Sun-Moon elongation
    psi = sun.separation(moon).to(u.rad).value

    # Illuminated fraction
    fli = (1 - np.cos(psi)) / 2
    return fli, alt

# ───────────────────────────────────────────────────────────────
# Load CSV with FITS info
# ───────────────────────────────────────────────────────────────
csv_file = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\data 2\Rezaei_Altafi_10_07_2025\background_Moon\moon-info.csv"
df = pd.read_csv(csv_file, dtype={'Negative DEC': bool}, converters={'RA': str, 'DEC': str})
df['RA']  = df['RA'].apply(timedelta_to_hms)
df['DEC'] = df['DEC'].apply(timedelta_to_hms)

# ───────────────────────────────────────────────────────────────
# Compute target alt, moon separation, moon alt, illuminated fraction
# ───────────────────────────────────────────────────────────────
results = []

for idx, row in df.iterrows():
    fits_path = row['file']
    ra_str = row['RA'].strip()
    dec_str = row['DEC'].strip()

    if row['Negative DEC'] and not dec_str.startswith('-'):
        dec_str = f"-{dec_str}"

    if len(ra_str.split(':')) != 3 or len(dec_str.split(':')) != 3:
        print(f"Skipping row {idx} due to invalid RA/DEC -> {ra_str}, {dec_str}")
        continue

    # Read FITS header (UTC)
    try:
        with fits.open(fits_path) as hdul:
            utc_time_str = hdul[0].header['DATE']  # ISO format
    except Exception as e:
        print(f"Skipping row {idx} due to FITS read error: {e}")
        continue

    t_utc = Time(utc_time_str, scale='utc', location=SITE_LOCATION)

    sep_computed, alt_computed = moon_separation(t_utc, ra_str, dec_str)
    fli, moon_alt = get_fli(t_utc)

    results.append({
        'file': fits_path,
        'RA_used': ra_str,
        'DEC_used': dec_str,
        'alt_computed': alt_computed,
        'sep_computed': sep_computed,
        'moon_alt': moon_alt,
        'fli': fli
    })

df_results = pd.DataFrame(results)
df_results.to_csv("moon_alt_sep_verification_with_RA_DEC.csv", index=False)
print(df_results.head())

# ───────────────────────────────────────────────────────────────
# Load sky box statistics (x_start/y_start/width/height assumed)
# ───────────────────────────────────────────────────────────────
stats_list = []
BOX_WIDTH, BOX_HEIGHT = 100, 100  # default if missing

for idx, row in df_results.iterrows():
    fits_path = row['file']
    x_start = int(row.get('x_start', 0))
    y_start = int(row.get('y_start', 0))
    width   = int(row.get('width', BOX_WIDTH))
    height  = int(row.get('height', BOX_HEIGHT))

    with fits.open(fits_path) as hdul:
        data = hdul[0].data.astype(float)

    box = data[y_start:y_start+height, x_start:x_start+width]
    stats_list.append({
        'file': fits_path,
        'mean': np.mean(box),
        'median': np.median(box),
        'std': np.std(box),
        'min': np.min(box),
        'max': np.max(box),
        'n_pixels': box.size
    })

df_stats = pd.DataFrame(stats_list)
df_stats.to_csv("sky_box_statistics.csv", index=False)

# ───────────────────────────────────────────────────────────────
# Merge results and build 2D sky model
# ───────────────────────────────────────────────────────────────
df = pd.merge(df_results, df_stats, on='file')

sep  = df['sep_computed'].values
altT = df['alt_computed'].values
altM = df['moon_alt'].values
y    = df['median'].values

# ───────────────────────────────────────────────────────────────
# Sky brightness model
# ───────────────────────────────────────────────────────────────
def sky_model(coords, a, b, c):
    sep, alt_target, alt_moon = coords
    alt_t_rad = np.radians(alt_target)
    alt_m_rad = np.radians(alt_moon)

    # Airmass
    X_t = 1/np.sin(np.clip(alt_t_rad, 1e-3, None))
    X_m = 1/np.sin(np.clip(alt_m_rad, 1e-3, None))

    k = 0.2  # extinction
    extinction_factor = 10 ** (-0.4 * k * X_t)

    B = a * np.exp(-b * sep) * X_t * X_m * extinction_factor + c
    return B

# ───────────────────────────────────────────────────────────────
# Use given fit parameters
# ───────────────────────────────────────────────────────────────
a_fit, b_fit, c_fit = 6245, 0.063, 668
B_model = sky_model((sep, altT, altM), a_fit, b_fit, c_fit)

# ───────────────────────────────────────────────────────────────
# Plot measured vs separation, color-coded by Moon altitude
# ───────────────────────────────────────────────────────────────
plt.figure(figsize=(10,6))
sc = plt.scatter(sep, y, c=altM, cmap='plasma', alpha=0.85, label='Measured medians')
plt.plot(sep, B_model, 'g-', lw=2, label='Sky model (2D)')
plt.xlabel("Moon–Target Separation (deg)")
plt.ylabel("Sky Background Median (ADU)")
plt.title("Sky Background vs Moon Separation (2D Model)")
plt.colorbar(sc, label="Moon Altitude (deg)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
