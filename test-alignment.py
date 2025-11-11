import os
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from astropy.nddata.utils import Cutout2D

# ---------------- User Settings ---------------- #
folder = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\Rezaei_Hossein_Atanaz_Kosar_2025_11_04\light\Aysan\high\hot pixels removed\97b-8\aligned\i"
cutout_size = 50  # pixels around the star
# ------------------------------------------------ #

# Get all FITS files in folder
fits_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.fit')]
fits_files.sort()

# Load first image to select stars
ref_file = fits_files[0]
with fits.open(ref_file) as hdul:
    data = hdul[0].data
    wcs = WCS(hdul[0].header)

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.imshow(data, cmap='gray', origin='lower', vmin=np.percentile(data, 5), vmax=np.percentile(data, 95))
ax.set_title("Click on a star, then close the window")
coords = []

def onclick(event):
    if event.inaxes == ax:
        x, y = event.xdata, event.ydata
        coords.append((x, y))
        ax.plot(x, y, 'rx')
        fig.canvas.draw()

cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

print("Selected pixel coordinates:", coords)

# Loop over all files and show cutouts, check RA/Dec
for f in fits_files:
    with fits.open(f) as hdul:
        data = hdul[0].data
        wcs = WCS(hdul[0].header)
        print(f"\nFile: {f}")
        for i, (x, y) in enumerate(coords):
            cutout = Cutout2D(data, (x, y), size=cutout_size)
            cutout_data = cutout.data
            ra, dec = wcs.wcs_pix2world(x, y, 0)
            print(f"Star {i+1}: RA={ra:.6f}, Dec={dec:.6f}")
            
            plt.figure()
            plt.imshow(cutout_data, cmap='gray', origin='lower')
            plt.title(f"{os.path.basename(f)} - Star {i+1}")
            plt.show()
