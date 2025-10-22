from astropy.io import fits
from astropy.visualization import (MinMaxInterval, LinearStretch, ImageNormalize)
import numpy as np
import matplotlib.pyplot as plt

# Load your FITS master frames
g_data = fits.getdata(r"C:\\Users\\AYSAN\\Desktop\\project\\INO\\ETC\\Outputs\\reduced and aligned\\crab\\g high\\crab_g_master.fits")
r_data = fits.getdata(r"C:\\Users\\AYSAN\\Desktop\\project\\INO\\ETC\\Outputs\\reduced and aligned\\crab\\r high\\crab_r_master.fits")
i_data = fits.getdata(r"C:\\Users\\AYSAN\\Desktop\\project\\INO\\ETC\\Outputs\\reduced and aligned\\crab\\i high\\crab_i_master.fits")

# Normalize and stretch each channel
norm = ImageNormalize(stretch=LinearStretch(), interval=MinMaxInterval())

r_norm = norm(r_data)
g_norm = norm(g_data)
i_norm = norm(i_data)

# Stack into RGB image
rgb = np.dstack((g_norm, r_norm, i_norm))
rgb = np.clip(rgb, 0, 1)

# Crop region of interest
crop = rgb[200:2000, 200:2000]

# --- Display ---
plt.figure(figsize=(8, 8))
plt.imshow(crop, origin='lower')
plt.axis('off')
plt.title("Cropped Colorized Crab Nebula (g, r, i bands)")
plt.show()

# --- Save with highest quality ---
output_path = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\crab_color_highres.png"
plt.imsave(output_path, crop, origin='lower', dpi=600, format='png')  # lossless and high DPI
print(f"Saved high-quality image to: {output_path}")
