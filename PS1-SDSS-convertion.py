import pandas as pd

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

# ---------------- Main code ----------------

# Path to your existing PS1 CSV
csv_path = r"C:\Users\AYSAN\Desktop\project\INO\ETC\PAN-star\97b-8_compact_top100.csv"

# Read the CSV
ps1_table = pd.read_csv(csv_path)

# Compute SDSS magnitudes
sdss_mags = ps1_to_sdss(ps1_table)

# Add SDSS mags to the existing DataFrame
ps1_table[['g_sdss', 'r_sdss', 'i_sdss']] = sdss_mags

# Save back to the same CSV (overwrite)
ps1_table.to_csv(csv_path, index=False)

print(f"Updated CSV with SDSS magnitudes saved at: {csv_path}")
