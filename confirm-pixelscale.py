from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

# Replace with one of your WCS-enabled FITS files
fits_file = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\Rezaei_Hossein_Atanaz_Kosar_2025_11_04\light\Aysan\high\hot pixels removed\97b-8\aligned\i\aligned_97b_8_i_2025_11_05_1x1_exp00.00.01.000_000001_High_2_cycleclean_iter3.fit"

# Open FITS and WCS
with fits.open(fits_file) as hdul:
    header = hdul[0].header
    wcs = WCS(header)

# Pixel scale calculation
# Get CD or CDELT matrix
if wcs.wcs.has_cd():
    cd = wcs.wcs.cd
    # Pixel scale in degrees
    scale_x = np.sqrt(cd[0,0]**2 + cd[0,1]**2)
    scale_y = np.sqrt(cd[1,0]**2 + cd[1,1]**2)
elif wcs.wcs.has_pc():
    # Use CDELT with PC matrix
    pc = wcs.wcs.pc
    cdelt = wcs.wcs.cdelt
    scale_x = np.sqrt((pc[0,0]*cdelt[0])**2 + (pc[0,1]*cdelt[1])**2)
    scale_y = np.sqrt((pc[1,0]*cdelt[0])**2 + (pc[1,1]*cdelt[1])**2)
else:
    # Simple CDELT
    scale_x = wcs.wcs.cdelt[0]
    scale_y = wcs.wcs.cdelt[1]

# Convert degrees to arcseconds
scale_x_arcsec = scale_x * 3600
scale_y_arcsec = scale_y * 3600

print(f"Pixel scale: {scale_x_arcsec:.3f}\" x {scale_y_arcsec:.3f}\" per pixel")
