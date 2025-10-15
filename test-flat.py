import numpy as np
from astropy.io import fits
import os
import matplotlib.pyplot as plt
# Hardcoded paths
masterflat_path = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\masterframes\Sept 30 masterflat\good\masterflat_g_0.29069s_2025-09-30_bin2x2_HIGH.fits"
masterdark_path = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\masterframes\Oct 1 masterdarks\masterdark_0.29069s_2025-10-01_bin2x2_HIGH.fits"
science_path    = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\Sep30\target3\g\high\area95_g_T10C_2025_10_02_2x2_exp00.01.00.000_000001_High_1.fit"
output_path     = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs"

# Load master dark
with fits.open(masterdark_path) as hd:
    md_data = hd[0].data.astype(float)
    md_hdr  = hd[0].header

# Load master flat
with fits.open(masterflat_path) as hf:
    mf_data = hf[0].data.astype(float)
    mf_hdr  = hf[0].header


gain_table_dark = (mf_data-md_data) / np.nanmedian(mf_data-md_data)
plt.imshow(gain_table_dark)
plt.title("gain_table_dark")
plt.show()
plt.hist(gain_table_dark.ravel(), bins=100, color='steelblue')
plt.title("gain_table_dark")
plt.show()
print(np.min(gain_table_dark), np.max(gain_table_dark))


gain_table = (mf_data) / np.nanmedian(mf_data)
plt.imshow(gain_table)
plt.title("gain_table")
plt.show()
plt.hist(gain_table.ravel(), bins=100, color='steelblue')
plt.title("gain_table")
plt.show()
print(np.min(gain_table), np.max(gain_table))

# Load science image
with fits.open(science_path) as hs:
    raw_data = hs[0].data.astype(float)
    raw_hdr  = hs[0].header

# Check shape compatibility
if raw_data.shape != md_data.shape or raw_data.shape != gain_table.shape:
    raise ValueError("Shape mismatch between science, dark, and flat frames.")

# Apply correction
corrected = (raw_data - md_data) / gain_table

