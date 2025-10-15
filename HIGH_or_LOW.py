from astropy.io import fits
from astropy.visualization import ZScaleInterval
import numpy as np
import matplotlib.pyplot as plt

# Load FITS files
low_path = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\Sep30\Rezaei_30_sep_2025\target3\g\target3_g_T10C_2025_10_01_2x2_exp00.01.00.000_000001_LOW_4.fit"
high_path = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\Sep30\Rezaei_30_sep_2025\target3\g\high\1 min\target3_g_T10C_2025_10_01_2x2_exp00.01.00.000_000001_High_4.fit"

with fits.open(low_path) as hdul1:
    low = hdul1[0].data

with fits.open(high_path) as hdul2:
    high = hdul2[0].data

# Compute gain table
gain_table = high / low



from astropy.visualization import ZScaleInterval, ImageNormalize

# Compute z-scale normalization
interval = ZScaleInterval()
norm = ImageNormalize(gain_table, interval=interval)

# Plot using norm
plt.imshow(gain_table, origin="lower", cmap="gray",interpolation=None)
plt.colorbar(label='Gain')
plt.title("Gain Table")
plt.show()

# STAR

x_star, y_star = 793, 893
cutout_size = 15

# Extract cutouts
x1, x2 = x_star - cutout_size, x_star + cutout_size
y1, y2 = y_star - cutout_size, y_star + cutout_size

low_cutout = low[y1:y2, x1:x2]
high_cutout = high[y1:y2, x1:x2]
gain_cutout = gain_table[y1:y2, x1:x2]

# Z-scale limits
interval = ZScaleInterval()
vmin_low, vmax_low = interval.get_limits(low_cutout)
vmin_high, vmax_high = interval.get_limits(high_cutout)
vmin_gain, vmax_gain = interval.get_limits(gain_cutout)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

im1 = axes[0].imshow(low_cutout, origin='lower', cmap='gray')
axes[0].set_title("Low Image Cutout")
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(high_cutout, origin='lower', cmap='gray')
axes[1].set_title("High Image Cutout")
plt.colorbar(im2, ax=axes[1])

im3 = axes[2].imshow(gain_cutout, origin='lower', cmap='gray')
axes[2].set_title("Gain Table Cutout")
plt.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Plot annotated gain cutout without z-scale
fig2, ax = plt.subplots(figsize=(6, 6))

im = ax.imshow(gain_cutout, origin='lower', cmap='viridis', interpolation=None)
plt.colorbar(im, ax=ax, label='Gain')

ax.set_title("Annotated Gain Table Cutout")

# Annotate each pixel with its value (1 decimal)
for (i, j), val in np.ndenumerate(gain_cutout):
    ax.text(j, i, f"{val:.0f}", ha='center', va='center', color='black', fontsize=8)

plt.tight_layout()
plt.show()