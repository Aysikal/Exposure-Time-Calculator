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

#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════#
# Flexible values: 
box_size = 300  # Size of the star box 
seeing = 1.3  # arcseconds
aperture_ratio = 1.5
pixel_scale = 0.047  # arcseconds per pixel
color = 'YlGn' # r filter: "Oranges", g filter : "Greens", u filter = "Purples", i filter : "Reds"
folder_path = r"C:\Users\AYSAN\Desktop\project\INO\gd246\g\high"  # Location of your images
star_coordinates_loc = r"C:\Users\AYSAN\Desktop\project\INO\star coordinates\Star_Coordinates_for_Green_Filter_High_gd246.npy"  # Location of the coordinates
#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════#
r_in_ratio = 2.4
r_out_ratio = 3
star_coordinates = np.load(star_coordinates_loc)
#print(star_coordinates)

seeing_pixels = seeing / pixel_scale
aperture_radius = aperture_ratio * seeing_pixels  # Aperture radius in pixels

def open_fits(path):
    with fits.open(path) as fitsfile:
        data = fitsfile[0].data
    return data

star_boxes = []
com_positions_in_boxes = []  # COM positions within each box
star_flux = []
sky_flux = []

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
    
    # Adjust the COM position within the box
    com_y = star_coordinates[idx][0] - y_start
    com_x = star_coordinates[idx][1] - x_start
    
    # Store the box and the COM position
    star_boxes.append(box)
    com_positions_in_boxes.append((com_x, com_y))
    
    # Calculate flux within the aperture
    positions = [(com_x, com_y)]
    apertures = CircularAperture(positions, r=aperture_radius)
    aperture_phot = aperture_photometry(box, apertures)
    star_flux.append(aperture_phot['aperture_sum'][0])
    
    # Calculate flux within the annulus
    r_in = aperture_radius * r_in_ratio
    r_out = aperture_radius * r_out_ratio
    annulus_apertures = CircularAnnulus(positions, r_in=r_in, r_out=r_out)
    annulus_phot = aperture_photometry(box, annulus_apertures)
    sky_flux.append(annulus_phot['aperture_sum'][0])

# Determine the grid size for plotting
num_boxes = len(star_boxes)
grid_size = int(np.ceil(np.sqrt(num_boxes)))

# Create the plot
fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
axes = axes.flatten()

for idx, (ax, box) in enumerate(zip(axes, star_boxes)):
    # Get the COM position for the current box
    com_x, com_y = com_positions_in_boxes[idx]
    height, width = box.shape
    
    # Display the star box with appropriate scaling
    norm = simple_norm(box, 'sqrt', percent=99)  # Enhance visibility
    ax.imshow(box, cmap=color, origin='lower', norm=norm)
    ax.axis('off')
    ax.set_title(f"gd246 High {idx+1}", fontsize=8)
    
    # Plot the aperture circle using CircularAperture
    positions = [(com_x, com_y)]
    apertures = CircularAperture(positions, r=aperture_radius)
    apertures.plot(color='red', lw=1, alpha=0.6, ax=ax)
    
    # Plot the annulus using CircularAnnulus
    r_in = aperture_radius * r_in_ratio
    r_out = aperture_radius * r_out_ratio
    annulus_apertures = CircularAnnulus(positions, r_in=r_in, r_out=r_out)
    annulus_apertures.plot(color='blue', lw=1, alpha=0.6, ax=ax)

# Hide any empty subplots
for ax in axes[num_boxes:]:
    ax.axis('off')

plt.tight_layout()
plt.show()

# Print the flux lists
print("Mean Star Flux:", np.mean(star_flux))
print("Mean Sky Flux:", np.mean(sky_flux))
print("Star Flux Minus Sky Flux" , np.mean(star_flux) - np.mean(sky_flux))

