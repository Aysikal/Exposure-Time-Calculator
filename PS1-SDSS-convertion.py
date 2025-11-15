import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

def ps1_to_sdss(ps1_df):
    """
    Convert PS1 g, r, i magnitudes to SDSS g', r', i' magnitudes.
    """
    g_ps1 = ps1_df['g_mag']
    r_ps1 = ps1_df['r_mag']
    i_ps1 = ps1_df['i_mag']
    
    # Compute colors
    g_r = g_ps1 - r_ps1
    r_i = r_ps1 - i_ps1
    i_z = i_ps1 - ps1_df.get('z_mag', pd.Series([0]*len(ps1_df)))  # optional z-mag
    
    # Apply approximate transformations (Tonry+ 2012)
    g_sdss = g_ps1 + 0.013 + 0.145 * g_r
    r_sdss = r_ps1 + 0.001 + 0.003 * r_i
    i_sdss = i_ps1 + 0.003 + 0.008 * i_z
    
    return pd.DataFrame({
        'g_sdss': g_sdss,
        'r_sdss': r_sdss,
        'i_sdss': i_sdss
    })

def convert_ra_dec_to_sexagesimal(df):
    """
    Convert RA/Dec in degrees to sexagesimal format (hh:mm:ss.ss and ±dd:mm:ss.ss).
    """
    coords = SkyCoord(ra=df['ra'].values * u.deg, dec=df['dec'].values * u.deg)
    df['ra_hms'] = coords.ra.to_string(unit=u.hour, sep=':', pad=True, precision=2)
    df['dec_dms'] = coords.dec.to_string(unit=u.deg, sep=':', alwayssign=True, pad=True, precision=2)
    return df

# ---------------- Main code ----------------

# Path to your existing PS1 CSV
csv_path = r"C:\Users\AYSAN\Desktop\project\INO\ETC\PAN-star\97b-8_compact_top50.csv"

# Read the CSV
ps1_table = pd.read_csv(csv_path)

# Compute SDSS magnitudes
sdss_mags = ps1_to_sdss(ps1_table)
ps1_table[['g_sdss', 'r_sdss', 'i_sdss']] = sdss_mags

# Convert RA/Dec to sexagesimal format
ps1_table = convert_ra_dec_to_sexagesimal(ps1_table)

# Save back to the same CSV (overwrite)
ps1_table.to_csv(csv_path, index=False)

print(f"Updated CSV with SDSS magnitudes and formatted RA/Dec saved at: {csv_path}")