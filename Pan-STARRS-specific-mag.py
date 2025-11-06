from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
from astroquery.mast import Catalogs
import pandas as pd
import numpy as np
import os

# -------------------------------------------------------
# Load FITS
# -------------------------------------------------------
fits_path = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\rezaei_saba_farideH_2025_10_22\GRB251013c\new-image.fits"
hdu = fits.open(fits_path)[0]
data = hdu.data
wcs = WCS(hdu.header)
ny, nx = data.shape  # image dimensions

# -------------------------------------------------------
# Query Pan-STARRS around field center
# -------------------------------------------------------
ra_center = 345.818376
dec_center = -0.2032108
radius = 0.1 * u.deg       # ~6 arcmin search radius

center = SkyCoord(ra_center, dec_center, unit=(u.deg, u.deg))
ps = Catalogs.query_region(center, radius=radius, catalog="Panstarrs")
ps_df = ps.to_pandas()

# Keep stars that have an i-band magnitude only
ps_df_filtered = ps_df.dropna(subset=['iMeanPSFMag']).copy()

# Only faint stars (i > 22.2)
ps_df_filtered = ps_df_filtered[ps_df_filtered['iMeanPSFMag'] > 22.2].copy()

# Convert RA/Dec → pixel coordinates
# -------------------------------------------------------
coords = SkyCoord(ps_df_filtered['raMean'], ps_df_filtered['decMean'], unit='deg')
xpix, ypix = wcs.world_to_pixel(coords)

# -------------------------------------------------------
# Keep only stars that fall inside the image
# -------------------------------------------------------
mask = (xpix >= 0) & (xpix < nx) & (ypix >= 0) & (ypix < ny)
ps_df_on_image = ps_df_filtered[mask].copy()
xpix_on_image = xpix[mask]
ypix_on_image = ypix[mask]

# -------------------------------------------------------
# Build compact output table
# -------------------------------------------------------
output_table = pd.DataFrame({
    'ID': range(len(ps_df_on_image)),
    'ra': ps_df_on_image['raMean'],
    'dec': ps_df_on_image['decMean'],
    'g_mag': ps_df_on_image['gMeanPSFMag'],
    'r_mag': ps_df_on_image['rMeanPSFMag'],
    'i_mag': ps_df_on_image['iMeanPSFMag']
})

print("\nSelected faint stars (i > 22.2):")
print(output_table)

# -------------------------------------------------------
# Save compact CSV
# -------------------------------------------------------
output_dir = r"C:\Users\AYSAN\Desktop\project\INO\ETC\PAN-star"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "GRB_stars_compact_faint_i_gt_22.2.csv")
output_table.to_csv(output_path, index=False)

print(f"\nSaved faint star table to:\n{output_path}")

# -------------------------------------------------------
# Plot FITS image + faint stars
# -------------------------------------------------------
plt.figure(figsize=(8,8))
plt.imshow(data, origin='lower', cmap='gray',
           vmin=np.percentile(data, 5),
           vmax=np.percentile(data, 99))

# star markers
plt.scatter(xpix_on_image, ypix_on_image,
            s=50/(ps_df_on_image['rMeanPSFMag']+0.1),
            edgecolor='red', facecolor='none', lw=1.0, zorder=5)

# labels
for idx, (x, y) in enumerate(zip(xpix_on_image, ypix_on_image)):
    plt.text(x+3, y+3, str(idx), color='yellow', fontsize=8)

plt.xlabel("X [pixels]")
plt.ylabel("Y [pixels]")
plt.title("Pan-STARRS stars with i > 22.2 on FITS image")
plt.tight_layout()
plt.show()
