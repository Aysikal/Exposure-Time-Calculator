from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
from astroquery.mast import Catalogs
import pandas as pd
import numpy as np
import os

# ----------- Load FITS ----------
fits_path = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\Rezaei_Hossein_Atanaz_Kosar_2025_11_04\light\Aysan\high\hot pixels removed\97b-8\aligned\g\dark_corrected\aligned_97b_8_g_2025_11_05_1x1_exp00.00.01.000_000001_High_2_cycleclean_iter3_dark_corrected.fit"
hdu = fits.open(fits_path)[0]  # first HDU
data = hdu.data
wcs = WCS(hdu.header)
ny, nx = data.shape  # image dimensions

# ----------- Query Pan-STARRS ----------
# Correct center coordinates (decimal degrees)
ra_center = 89.604   # 05:58:25.03
dec_center = 0.087   # +00:05:13.56
radius = 0.1 * u.deg   # a slightly larger search radius

center = SkyCoord(ra_center, dec_center, unit='deg')
  
ps = Catalogs.query_region(center, radius=radius, catalog="Panstarrs")
ps_df = ps.to_pandas()

# Filter stars with g', r', i' available
ps_df_filtered = ps_df.dropna(subset=['gMeanPSFMag','rMeanPSFMag','iMeanPSFMag']).copy()

# ----------- Convert RA/Dec to pixel coordinates ----------
coords = SkyCoord(ps_df_filtered['raMean'], ps_df_filtered['decMean'], unit='deg')
xpix, ypix = wcs.world_to_pixel(coords)

# add pixel columns safely
ps_df_filtered = ps_df_filtered.assign(xpix=xpix, ypix=ypix)

# ----------- Keep only stars inside the image ----------
mask = (ps_df_filtered['xpix'] >= 0) & (ps_df_filtered['xpix'] < nx) & (ps_df_filtered['ypix'] >= 0) & (ps_df_filtered['ypix'] < ny)
ps_df_on_image = ps_df_filtered.loc[mask].copy()

# ----------- Select the 100 brightest (by r-band) ----------
# change 'rMeanPSFMag' to 'gMeanPSFMag' or 'iMeanPSFMag' if you prefer another band
ps_df_on_image.sort_values('rMeanPSFMag', inplace=True)  # ascending -> brightest first
top_n = 50
ps_top = ps_df_on_image.head(top_n).reset_index(drop=True)

xpix_on_image = ps_top['xpix'].values
ypix_on_image = ps_top['ypix'].values

# ----------- Create compact table for output ----------
output_table = pd.DataFrame({
    'ID': np.arange(1, len(ps_top) + 1),  # 1-based ID for plotting
    'ra': ps_top['raMean'],
    'dec': ps_top['decMean'],
    'g_mag': ps_top['gMeanPSFMag'],
    'r_mag': ps_top['rMeanPSFMag'],
    'i_mag': ps_top['iMeanPSFMag']
})

print(output_table)

# ----------- Save the table ----------
output_dir = r"C:\Users\AYSAN\Desktop\project\INO\ETC\PAN-star"
os.makedirs(output_dir, exist_ok=True)  # make sure directory exists
output_path = os.path.join(output_dir, "97b-8_compact_top100.csv")
output_table.to_csv(output_path, index=False)
print(f"Saved compact table to: {output_path}")

# ----------- Plot FITS image and overlay stars ----------
plt.figure(figsize=(8,8))
plt.imshow(data, origin='lower', cmap='gray', vmin=np.percentile(data, 5), vmax=np.percentile(data, 99))

# compute marker sizes and clip to reasonable range
sizes = np.clip(50 / (ps_top['rMeanPSFMag'].values + 0.1), 5, 200)

plt.scatter(xpix_on_image, ypix_on_image, s=sizes,
            edgecolor='red', facecolor='none', lw=1.0, zorder=5)

# Label stars with their ID (1..N)
for idx, (x, y) in enumerate(zip(xpix_on_image, ypix_on_image), start=1):
    plt.text(x+3, y+3, str(idx), color='black', fontsize=8)

plt.xlabel('X [pixels]')
plt.ylabel('Y [pixels]')
plt.title("Top 100 Pan-STARRS stars on FITS image (by r-band brightness)")
plt.tight_layout()
plt.show()