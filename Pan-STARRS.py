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
fits_path = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\rezaei_saba_farideH_2025_10_22\GRB251013c\new-image.fits"
hdu = fits.open(fits_path)[0]  # first HDU
data = hdu.data
wcs = WCS(hdu.header)
ny, nx = data.shape  # image dimensions

# ----------- Query Pan-STARRS ----------
ra_center = 345.818376
dec_center = -0.2032108
radius = 0.1 * u.deg  # increased radius (~6 arcmin)
center = SkyCoord(ra_center, dec_center, unit=(u.deg, u.deg))

ps = Catalogs.query_region(center, radius=radius, catalog="Panstarrs")
ps_df = ps.to_pandas()

# Filter stars with g', r', i' available
ps_df_filtered = ps_df.dropna(subset=['gMeanPSFMag','rMeanPSFMag','iMeanPSFMag']).copy()

# ----------- Convert RA/Dec to pixel coordinates ----------
coords = SkyCoord(ps_df_filtered['raMean'], ps_df_filtered['decMean'], unit='deg')
xpix, ypix = wcs.world_to_pixel(coords)

# ----------- Keep only stars inside the image ----------
mask = (xpix >= 0) & (xpix < nx) & (ypix >= 0) & (ypix < ny)
ps_df_on_image = ps_df_filtered[mask].copy()
xpix_on_image = xpix[mask]
ypix_on_image = ypix[mask]

# ----------- Create compact table for output ----------
output_table = pd.DataFrame({
    'ID': range(len(ps_df_on_image)),  # number on the plot
    'ra': ps_df_on_image['raMean'],
    'dec': ps_df_on_image['decMean'],
    'g_mag': ps_df_on_image['gMeanPSFMag'],
    'r_mag': ps_df_on_image['rMeanPSFMag'],
    'i_mag': ps_df_on_image['iMeanPSFMag']
})

print(output_table)

# ----------- Save the table ----------
output_dir = r"C:\Users\AYSAN\Desktop\project\INO\ETC\PAN-star"
os.makedirs(output_dir, exist_ok=True)  # make sure directory exists
output_path = os.path.join(output_dir, "GRB_stars_compact.csv")
output_table.to_csv(output_path, index=False)
print(f"Saved compact table to: {output_path}")

# ----------- Plot FITS image and overlay stars ----------
plt.figure(figsize=(8,8))
plt.imshow(data, origin='lower', cmap='gray', vmin=np.percentile(data, 5), vmax=np.percentile(data, 99))
plt.scatter(xpix_on_image, ypix_on_image, s=50/(ps_df_on_image['rMeanPSFMag']+0.1),
            edgecolor='red', facecolor='none', lw=1.0, zorder=5)

# Label stars with their row number (ID)
for idx, (x, y) in enumerate(zip(xpix_on_image, ypix_on_image)):
    plt.text(x+3, y+3, str(idx), color='black', fontsize=8)

plt.xlabel('X [pixels]')
plt.ylabel('Y [pixels]')
plt.title("Pan-STARRS stars on FITS image")
plt.tight_layout()
plt.show()
