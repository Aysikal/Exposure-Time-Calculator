#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════#
#This code was written by Aysan Hemmatiortakand. Last updated 9/30/2025 
#you can contact me for any additional questions or information via Email 
#email address :aysanhemmatiortakand@gmail.com
#github = https://github.com/Aysikal
#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════#

import os
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from astropy.visualization import ImageNormalize, ZScaleInterval
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════#
#input values:
folder_path = r"C:\Users\AYSAN\Desktop\project\INO\gd246\Test folder"  # location to where the images are located
save_directory = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\Star Coords"  # location to where you want to save the star coordinates
save_filename = "This is a test"  # The name you want to give the star coordinates file
#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════#
def open_fits(path):
    with fits.open(path) as fitsfile:
        file = fitsfile[0].data
        return file

def zscale_plot_with_magnifier(image_data, plot_title, colorbar_title):
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(image_data)
    norm = ImageNormalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(image_data, origin="lower", norm=norm, cmap='gray_r')  
    ax.set_title(plot_title)
    cbar = fig.colorbar(im)
    cbar.set_label(colorbar_title)

    # Create a magnifier inset
    zoom_factor = 5  # Increased zoom factor for larger magnification
    axins_size = 2  # Size of the inset axes
    axins = zoomed_inset_axes(ax, zoom=zoom_factor, loc='upper right', borderpad=1)
    axins.imshow(image_data, origin="lower", norm=norm, cmap='viridis')
    axins.set_xlim(0, 1)
    axins.set_ylim(0, 1)
    axins.axis('off')  # Hide axes ticks and labels

    # Rectangle to indicate magnified area in the main plot
    rect_size = 100  # Increase this value for a larger magnified area
    rect = Rectangle((0, 0), rect_size, rect_size, edgecolor='red', facecolor='none', linewidth=1)
    ax.add_patch(rect)

    def on_mouse_move(event):
        if event.inaxes == ax:
            xdata, ydata = event.xdata, event.ydata
            if xdata is not None and ydata is not None:
                x, y = int(xdata), int(ydata)
                # Define the size of the magnified region
                size = rect_size // 2
                x1 = max(x - size, 0)
                x2 = min(x + size, image_data.shape[1])
                y1 = max(y - size, 0)
                y2 = min(y + size, image_data.shape[0])

                # Update the inset axes limits
                axins.set_xlim(x1, x2)
                axins.set_ylim(y1, y2)
                axins.figure.canvas.draw_idle()

                # Update the rectangle position
                rect.set_xy((x1, y1))
                rect.set_width(x2 - x1)
                rect.set_height(y2 - y1)
                rect.figure.canvas.draw_idle()

    # Connect the mouse motion event to the on_mouse_move function
    motion_cid = fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

    return fig, ax

image_files = []
coordinates = []

# Collect all FITS files in the folder
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    image_files.append(file_path)

# Sort the files if necessary (optional)
image_files.sort()

# Function to handle mouse clicks
def onclick(event):
    x, y = event.xdata, event.ydata
    if x is not None and y is not None:
        coords = [y, x]  # [y, x] format
        coordinates.append(coords)
        plt.close()  # Close the figure to proceed to the next image
    else:
        print("Click inside the image area.")

# Loop through all images
for idx, file_path in enumerate(image_files):
    data = open_fits(file_path)

    # Display the image using the updated function
    plot_title = f'Image {idx+1}/{len(image_files)}: Click on reference star'
    colorbar_title = 'Pixel Intensity'
    fig, ax = zscale_plot_with_magnifier(data, plot_title, colorbar_title)

    # Connect the click event
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()

# Convert coordinates list to a numpy array
coordinates = np.array(coordinates)
print("Coordinates of selected points:")
print(coordinates)

save_path = os.path.join(save_directory, save_filename)

# Ensure the save directory exists
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Save the array as a .npy file
np.save(save_path, coordinates)
print(f"Coordinates array saved to {save_path}")
