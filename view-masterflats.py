import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# 📁 Path to your FITS folder
folder = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\masterframes\Sept 30 masterflat\masterflats from flats"
fits_files = sorted(glob.glob(os.path.join(folder, "*.fit*")))

# 🖼️ Plot in 4x4 grids
batch_size = 4
for i in range(0, len(fits_files), batch_size):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    for j, ax in enumerate(axes):
        idx = i + j
        if idx >= len(fits_files):
            ax.axis('off')
            continue

        file = fits_files[idx]
        with fits.open(file) as hdul:
            data = hdul[0].data

        ax.imshow(data, cmap='gray', origin='lower')
        ax.set_title(os.path.basename(file), fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    plt.show()