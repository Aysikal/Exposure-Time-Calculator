#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════#
#This code was written by Aysan Hemmatiortakand. Last updated 9/30/2025
#you can contact me for any additional questions or information via Email 
#email address :aysanhemmatiortakand@gmail.com
#github = https://github.com/Aysikal
#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════#

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from astropy.visualization import simple_norm
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

# Define the Gaussian function
def gaussian(x, A, mu, sigma):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════#
# Flexible values: 
box_size = 200  # Size of the star box 
pixel_scale = 0.047  # arcseconds per pixel
color = "Purples"  # r filter: "Oranges", g filter : "Greens", u filter = "Purples", i filter : "Reds"
filter = "i"  # choose filter color
mode = "Low"  # choose gain mode high/low
folder_path = r"C:\Users\AYSAN\Desktop\project\INO\gd246\g\high"  # Location of your images
star_coordinates_loc = r"C:\Users\AYSAN\Desktop\project\INO\star coordinates\Star_Coordinates_for_Green_Filter_High_gd246.npy"  # Location of the coordinates
specific_plot_idx = 0  # Index of the specific plot to display separately (0-based index)
#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════#

star_coordinates = np.load(star_coordinates_loc)

def open_fits(path):
    with fits.open(path) as fitsfile:
        data = fitsfile[0].data
    return data

def calculate_radial_profile(data, center, max_radius):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(int)  # Use int instead of np.int

    # Mask for distances greater than max_radius
    mask = r <= max_radius
    
    tbin = np.bincount(r[mask].ravel(), data[mask].ravel())
    nr = np.bincount(r[mask].ravel())
    
    radialprofile = np.zeros_like(tbin, dtype=float)
    radialprofile[nr > 0] = tbin[nr > 0] / nr[nr > 0]
    radialprofile = gaussian_filter1d(radialprofile, sigma=2)
    return radialprofile

def calculate_com(data):
    total = np.sum(data)
    y, x = np.indices(data.shape)
    com_y = np.sum(y * data) / total
    com_x = np.sum(x * data) / total
    return com_x, com_y

def calculate_hwhm(profile):
    half_max = (np.max(profile) + np.median(profile[-15:])) / 2.0
    closest_index = np.argmin(np.abs(profile - half_max))
    return closest_index

star_boxes = []
com_positions_in_boxes = []  # COM positions within each box
FWHM = []  # List to store FWHM values

for idx, filename in enumerate(os.listdir(folder_path)):
    file_path = os.path.join(folder_path, filename)
    data = open_fits(file_path)
    
    # Get approximate star center from provided coordinates
    y_center = int(round(star_coordinates[idx][0]))
    x_center = int(round(star_coordinates[idx][1]))
    
    # Box boundaries
    y_start = y_center - box_size // 2
    y_end = y_center + box_size // 2
    x_start = x_center - box_size // 2
    x_end = x_center + box_size // 2
    
    # Handle edge cases
    y_start = max(0, y_start)
    y_end = min(data.shape[0], y_end)
    x_start = max(0, x_start)
    x_end = min(data.shape[1], x_end)
    
    box = data[y_start:y_end, x_start:x_end]
    
    # Calculate COM within the box
    com_x, com_y = calculate_com(box)
    
    # Store the box and COM position
    star_boxes.append(box)
    com_positions_in_boxes.append((com_x, com_y))

# Calculate radial profiles
max_radius = box_size // 2
radial_profiles = [calculate_radial_profile(box, com, max_radius) for box, com in zip(star_boxes, com_positions_in_boxes)]

# Calculate HWHM for each radial profile
hwhms = [calculate_hwhm(profile) for profile in radial_profiles]
FWHM = [hwhm * 2 for hwhm in hwhms if not np.isnan(hwhm)]

# Print FWHMs
for idx, fwhm in enumerate(FWHM):
    print(f"Radial Profile {idx+1} FWHM: {fwhm:.2f} pixels")

# Determine the grid size for plotting
num_profiles = len(radial_profiles)
grid_size = int(np.ceil(np.sqrt(num_profiles)))

# Create the main plot
fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
axes = axes.flatten()

for idx, (ax, radial_profile) in enumerate(zip(axes, radial_profiles)):
    ax.plot(radial_profile, label='Radial Profile')
    
    half_max = (np.max(radial_profile) + np.median(radial_profile [-10:])) / 2.0
    ax.axhline(y=half_max, color='r', linestyle='--', label='Half-Maximum Line')
    
    com = com_positions_in_boxes[idx]
    ax.axvline(x=0, color='g', linestyle=':', label='COM Line')
    
    ax.set_ylabel('Intensity', fontsize=7)
    ax.set_title(f'Radial Profile {idx+1}\nHWHM: {hwhms[idx]:.2f} pixels', fontsize=7, pad=3)
    
    ax.tick_params(axis='both', which='major', labelsize=7, length=3, width=0.5)
    #ax.legend(fontsize=6)

# Hide any empty subplots
for ax in axes[num_profiles:]:
    ax.axis('off')

# Adjust the layout to give more space between plots
plt.subplots_adjust(hspace=0.5, wspace=0.5)

plt.tight_layout()
#plt.show()

# Plot the specified radial profile separately
fig, ax = plt.subplots(figsize=(8, 6))
radial_profile = radial_profiles[specific_plot_idx]
ax.plot(radial_profile, label='Radial Profile')

half_max = (np.max(radial_profile) + np.median(radial_profile [-10:]))/2
ax.axhline(y=half_max, color='r', linestyle='--', label='Half-Maximum Line')

com = com_positions_in_boxes[specific_plot_idx]
ax.axvline(x=0, color='g', linestyle=':', label='COM Line')

ax.set_ylabel('Intensity', fontsize=12)
ax.set_xlabel('Radius From COM (pixels)', fontsize=12)
ax.set_title(f'Radial Profile {specific_plot_idx+1}\nHWHM: {hwhms[specific_plot_idx]:.2f} pixels', fontsize=12, pad=10)
ax.tick_params(axis='both', which='major', labelsize=10, length=5, width=1)
ax.legend(fontsize=10)

plt.tight_layout()
#plt.show()

# Plot histogram of FWHM values 
fig, ax = plt.subplots(figsize=(8, 6))
counts, bin_edges = np.histogram(FWHM, bins=10)
# Calculate bin centers
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
ax.hist(FWHM, bins=10, edgecolor='k', alpha=0.6, label='Data')
ax.set_xlabel('FWHM (pixels)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Histogram of FWHM values', fontsize=12)
ax.legend(fontsize=10)
plt.tight_layout()
#plt.show()

# Plot histogram of FWHM values 
fig, ax = plt.subplots(figsize=(8, 6))
FWHM = np.array(FWHM)
FWHM_arcsec = FWHM * pixel_scale
counts, bin_edges = np.histogram(FWHM_arcsec, bins=10)
# Calculate bin centers
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
ax.hist(FWHM * pixel_scale, bins=10, edgecolor='k', alpha=0.6, label='Data')
ax.set_xlabel('FWHM (arcseconds)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Histogram of FWHM values', fontsize=12)
ax.legend(fontsize=10)

median_fwhm = np.mean(FWHM_arcsec)
ax.text(0.6, 0.95, f'Median seeing for gd246 {filter} filter ({mode}): {median_fwhm:.2f} arcseconds',
        verticalalignment='top', horizontalalignment='right',
        transform=ax.transAxes, fontsize=10)
plt.tight_layout()
#plt.show()

# Create the grid plot for the star boxes
fig_box, ax_box = plt.subplots(grid_size, grid_size, figsize=(15, 15))
axes_box = ax_box.flatten()

for idx, (ax, box) in enumerate(zip(axes_box, star_boxes)):
    norm = simple_norm(box, 'sqrt', percent=99)
    ax.imshow(box, origin='lower', cmap=color, norm=norm)
    ax.set_title(f'Star Box {idx+1}', fontsize=7, pad=3)
    ax.tick_params(axis='both', which='major', labelsize=7, length=3, width=0.5)
    # Remove axes ticks
    ax.set_xticks([])
    ax.set_yticks([])

# Hide any empty subplots
for ax in axes_box[num_profiles:]:
    ax.axis('off')

plt.tight_layout()
plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.show()
