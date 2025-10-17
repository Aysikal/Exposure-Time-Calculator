import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

def stretch(image, scale_min=None, scale_max=None):
    # Simple linear stretch with optional clipping
    if scale_min is None:
        scale_min = np.percentile(image, 1)
    if scale_max is None:
        scale_max = np.percentile(image, 99)
    stretched = np.clip((image - scale_min) / (scale_max - scale_min), 0, 1)
    return stretched

# Load FITS data
g_data = fits.getdata(r"C:\Users\AYSAN\Pictures\crab g high\crab_g_master.fits")
r_data = fits.getdata(r"C:\Users\AYSAN\Pictures\crab r high\crab_r_master.fits")
i_data = fits.getdata(r"C:\Users\AYSAN\Pictures\crab i high\crab_i_master.fits")

# Stretch each channel
R = stretch(i_data)
G = stretch(r_data)
B = stretch(g_data)

# Stack into RGB image
rgb = np.dstack((R, G, B))

# Display
plt.figure(figsize=(8, 8))
plt.imshow(rgb, origin='lower')
plt.axis('off')
plt.title('Crab Nebula (gri Composite)')
plt.show()
