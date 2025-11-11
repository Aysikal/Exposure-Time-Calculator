# ───────────────────────────────────────────────────────────────
# Verify Altitude and Moon-Target Separation from CSV files
# Author: Aysan Hemmatiortakand
# Last updated: 10/03/2025
# ───────────────────────────────────────────────────────────────

import pandas as pd
from astropy.io import fits
from astropy.coordinates import SkyCoord, AltAz, EarthLocation, get_body
from astropy.time import Time
from astroplan import Observer
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, get_body
import astropy.units as u

# ───────────────────────────────────────────────────────────────
# Observatory setup
# ───────────────────────────────────────────────────────────────
SITE_NAME     = "Iran National Observatory"
SITE_LAT      = 33.674
SITE_LON      = 51.3188
SITE_ELEV     = 3600
SITE_TIMEZONE = "Asia/Tehran"

SITE_LOCATION = EarthLocation(lat=SITE_LAT*u.deg, lon=SITE_LON*u.deg, height=SITE_ELEV*u.m)
SITE_OBSERVER = Observer(location=SITE_LOCATION, timezone=SITE_TIMEZONE, name=SITE_NAME)
EXPOSURE_TIME = 19
# ───────────────────────────────────────────────────────────────
# Helper: convert Timedelta or Excel time to 'H:M:S' string
# ───────────────────────────────────────────────────────────────
def timedelta_to_hms(td):
    """Convert pandas Timedelta or Excel time to 'H:M:S' string."""
    if isinstance(td, pd.Timedelta):
        total_seconds = td.total_seconds()
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60
        return f"{hours}:{minutes:02d}:{seconds:05.2f}"
    return str(td)

# ───────────────────────────────────────────────────────────────
# Function to compute Moon-target angular separation
# ───────────────────────────────────────────────────────────────
def moon_separation(date_str: str, hour: int, minute: int, ra: str, dec: str) -> float:
    t_utc = Time(f"{date_str} {hour:02d}:{minute:02d}", scale='utc', location=SITE_LOCATION)
    target_icrs = SkyCoord(ra=ra, dec=dec, unit=(u.hourangle, u.deg), frame="icrs")
    altaz_frame = AltAz(obstime=t_utc, location=SITE_LOCATION)
    target_altaz = target_icrs.transform_to(altaz_frame)
    moon_altaz = get_body("moon", t_utc, SITE_LOCATION).transform_to(altaz_frame)
    return target_altaz.separation(moon_altaz).degree

# ───────────────────────────────────────────────────────────────
# Read CSV
# ───────────────────────────────────────────────────────────────
csv_file = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\data 2\Rezaei_Altafi_10_07_2025\background_Moon\moon-info.csv"

df = pd.read_csv(
    csv_file,
    dtype={'Negative DEC': bool},
    converters={'RA': str, 'DEC': str}
)

df['RA']  = df['RA'].apply(timedelta_to_hms)
df['DEC'] = df['DEC'].apply(timedelta_to_hms)

# ───────────────────────────────────────────────────────────────
# Loop through FITS files
# ───────────────────────────────────────────────────────────────
results = []

for idx, row in df.iterrows():
    fits_path = row['file']
    ra_str = row['RA'].strip()
    dec_str = row['DEC'].strip()

    # Handle negative DEC
    if row['Negative DEC'] and not dec_str.startswith('-'):
        dec_str = f"-{dec_str}"

    # Validate format
    if len(ra_str.split(':')) != 3 or len(dec_str.split(':')) != 3:
        print(f"Skipping row {idx} due to invalid RA/DEC -> {ra_str}, {dec_str}")
        continue

    # Read FITS header
    try:
        with fits.open(fits_path) as hdul:
            utc_time_str = hdul[0].header['DATE']
    except Exception as e:
        print(f"Skipping row {idx} due to FITS read error: {e}")
        continue

    t_utc = Time(utc_time_str, scale='utc', location=SITE_LOCATION)
    altaz_frame = AltAz(obstime=t_utc, location=SITE_LOCATION)

    # Create SkyCoord
    try:
        coord = SkyCoord(ra=ra_str, dec=dec_str, unit=(u.hourangle, u.deg))
    except Exception as e:
        print(f"Skipping row {idx} due to SkyCoord error: {e}")
        continue

    # Compute altitude
    altaz = coord.transform_to(altaz_frame)
    alt_computed = altaz.alt.degree

    # Compute Moon separation
    sep_computed = moon_separation(
        date_str=t_utc.to_datetime().strftime("%Y-%m-%d"),
        hour=t_utc.to_datetime().hour,
        minute=t_utc.to_datetime().minute,
        ra=ra_str,
        dec=dec_str
    )

    alt_excel = row['Altitude']
    sep_excel = row['Seperation']

    results.append({
        'file': fits_path,
        'RA_used': ra_str,
        'DEC_used': dec_str,
        'alt_excel': alt_excel,
        'alt_computed': alt_computed,
        'alt_diff': alt_computed - alt_excel if alt_excel is not None else None,
        'sep_excel': sep_excel,
        'sep_computed': sep_computed,
        'sep_diff': sep_computed - sep_excel if sep_excel is not None else None
    })

# ───────────────────────────────────────────────────────────────
# Save and display results
# ───────────────────────────────────────────────────────────────
df_results = pd.DataFrame(results)
print(df_results[['file', 'RA_used', 'DEC_used', 'alt_computed', 'sep_computed', 'alt_diff', 'sep_diff']])
df_results.to_csv("moon_alt_sep_verification_with_RA_DEC.csv", index=False)


from astropy.visualization import ZScaleInterval, ImageNormalize

# CSV containing the FITS file paths
csv_file = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\data 2\Rezaei_Altafi_10_07_2025\background_Moon\moon-info.csv"
df = pd.read_csv(csv_file)
"""
# Preset fixed box size
BOX_WIDTH = 100
BOX_HEIGHT = 100

# Add empty columns for sky box info if not already present
for col in ["x_start", "y_start", "width", "height"]:
    if col not in df.columns:
        df[col] = np.nan

def on_click(event):
    
    if event.inaxes is None:
        return
    x_center = int(event.xdata)
    y_center = int(event.ydata)
    
    # Compute box start coordinates
    x_start = max(x_center - BOX_WIDTH//2, 0)
    y_start = max(y_center - BOX_HEIGHT//2, 0)
    
    # Update dataframe row with sky box info
    df.loc[idx, "x_start"] = x_start
    df.loc[idx, "y_start"] = y_start
    df.loc[idx, "width"]   = BOX_WIDTH
    df.loc[idx, "height"]  = BOX_HEIGHT
    
    # Draw rectangle on image for confirmation
    rect = Rectangle((x_start, y_start), BOX_WIDTH, BOX_HEIGHT,
                     edgecolor='red', facecolor='none', lw=2)
    ax.add_patch(rect)
    fig.canvas.draw()
    
    print(f"Sky box set for {current_file}: x={x_start}, y={y_start}")
    plt.close()

# Loop through each FITS file
for idx, row in df.iterrows():
    current_file = row['file']
    
    # Load FITS image
    with fits.open(current_file) as hdul:
        data = hdul[0].data
    
    # Use zscale normalization (robust against outliers)
    norm = ImageNormalize(data, interval=ZScaleInterval())
    
    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap='gray', origin='lower', norm=norm)
    ax.set_title(f"Click center of sky box for: {current_file}")
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Pixel value")
    
    # Connect click event
    cid = fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

# Overwrite the same CSV with updated sky box info
df.to_csv(csv_file, index=False)
print(f"Sky boxes saved back into {csv_file}")
"""
import pandas as pd
import numpy as np
from astropy.io import fits

# Load CSV with sky box coordinates
csv_file = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\data 2\Rezaei_Altafi_10_07_2025\background_Moon\moon-info.csv"
df = pd.read_csv(csv_file)

results = []

for idx, row in df.iterrows():
    current_file = row['file']
    x_start, y_start = int(row['x_start']), int(row['y_start'])
    width, height = int(row['width']), int(row['height'])
    
    # Load FITS image
    with fits.open(current_file) as hdul:
        data = hdul[0].data.astype(float)
    
    # Extract box region
    box = data[y_start:y_start+height, x_start:x_start+width]/EXPOSURE_TIME
    
    # Compute statistics
    stats = {
        'file': current_file,
        'mean': np.mean(box),
        'median': (np.median(box)),
        'std': np.std(box),
        'min': np.min(box),
        'max': np.max(box),
        'n_pixels': box.size,
        'box_sum' : np.sum(box)
    }
    results.append(stats)

# Save results
df_stats = pd.DataFrame(results)
df_stats.to_csv("sky_box_statistics.csv", index=False)
print(df_stats)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from astropy.io import fits
from astropy.stats import sigma_clip

# Load CSVs
df_sep = pd.read_csv("moon_alt_sep_verification_with_RA_DEC.csv")
df_stats = pd.read_csv("sky_box_statistics.csv")
df = pd.merge(df_sep, df_stats, on="file")

x = df['sep_computed'].values
y = df['median'].values

# Exponential fit
def exp_decay(x, a, b, c):
    return a * np.exp(-b * x) + c

params, _ = curve_fit(exp_decay, x, y, p0=[max(y), 0.05, min(y)])
a, b, c = params

# Plot scatter + fit
plt.figure(figsize=(8,6))
plt.scatter(x, y, label="Sigma-clipped medians", alpha=0.7)
plt.plot(x, exp_decay(x, *params), 'g-', label="Exponential fit")

plt.xlabel("Moon-Target Separation (degrees)")
plt.ylabel("Sky Box Median Value (sigma-clipped)")
plt.title("Sky Background vs Moon Separation (Sigma-clipped)")
plt.legend()
plt.grid(True)

# Add parameter text
param_text = (
    f"a = {a:.3f}\n"
    f"b = {b:.3f}\n"
    f"c = {c:.3f}"
)

plt.text(
    0.05, 0.95,
    param_text,
    transform=plt.gca().transAxes,
    fontsize=11,
    verticalalignment='top',
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
)

plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Constants
PIXEL_AREA = (0.101 * 2)**2  # arcsec^2
ZERO_POINT = 24.0316
EXPOSURE_TIME = 19  # seconds
EXTINCTION_COEFF = 0.2

# Load sky box statistics
stats_file = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Codes\sky_box_statistics.csv"
df_stats = pd.read_csv(stats_file)

# Compute flux per pixel per second
df_stats['flux_per_pixel_per_sec'] = df_stats['box_sum'] / (df_stats['n_pixels'] * EXPOSURE_TIME)

# Compute surface brightness
df_stats['mu_mag_arcsec2'] = ZERO_POINT - 2.5 * np.log10(df_stats['flux_per_pixel_per_sec'] / PIXEL_AREA)

# Print results
print(df_stats[['file', 'box_sum', 'n_pixels', 'flux_per_pixel_per_sec', 'mu_mag_arcsec2']])

# Save back to CSV
df_stats.to_csv(stats_file, index=False)
print(f"\n✅ Surface brightness values saved to: {stats_file}")

# Load moon separation and altitude data
df_sep = pd.read_csv("moon_alt_sep_verification_with_RA_DEC.csv")

# Merge with sky box stats
df = pd.merge(df_sep, df_stats[['file', 'mu_mag_arcsec2']], on="file")

# Compute airmass from altitude
alt_deg = df['alt_computed'].values
airmass = 1 / np.cos(np.radians(90 - alt_deg))

# Apply extinction correction
mu_observed = df['mu_mag_arcsec2'].values
mu_corrected = mu_observed - EXTINCTION_COEFF * airmass

# Add to DataFrame
df['airmass'] = airmass
df['mu_mag_arcsec2_corrected'] = mu_corrected

# Save updated CSV
df.to_csv("moon_alt_sep_verification_with_RA_DEC.csv", index=False)
print("\n✅ Extinction-corrected magnitudes saved to: moon_alt_sep_verification_with_RA_DEC.csv")
print(df[['file', 'mu_mag_arcsec2', 'airmass', 'mu_mag_arcsec2_corrected']].head())

# Fit exponential decay model
def exp_decay(x, a, b, c):
    return a * np.exp(-b * x) + c

x = df['sep_computed'].values
y = df['mu_mag_arcsec2_corrected'].values

params, _ = curve_fit(exp_decay, x, y, p0=[max(y), 0.05, min(y)])
a, b, c = params

# Plot scatter + fit
plt.figure(figsize=(8,6))
plt.scatter(x, y, alpha=0.7, label="Corrected Magnitudes")
plt.plot(x, exp_decay(x, *params), 'g-', label="Exponential fit")

plt.xlabel("Moon-Target Separation (degrees)")
plt.ylabel("Sky Box Magnitude (mag/arcsec²)")
plt.gca().invert_yaxis()
plt.title("Extinction-Corrected Sky Magnitude vs Moon Separation")
plt.legend()
plt.grid(True)

# Annotate fit parameters
param_text = f"a = {a:.3f}\nb = {b:.3f}\nc = {c:.3f}"
plt.text(0.05, 0.95, param_text, transform=plt.gca().transAxes,
         fontsize=11, verticalalignment='top',
         bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

plt.tight_layout()
plt.show()