#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════#
# This code was written by Aysan Hemmatiortakand. Last updated 9/30/2025
# You can contact me for any additional questions or information via Email
# Email address: aysanhemmatiortakand@gmail.com
# GitHub: https://github.com/Aysikal
#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════#

import os
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from astropy.visualization import ImageNormalize, ZScaleInterval
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes

#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════#
# Input values:
folder_path = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\Sep30\Rezaei_30_sep_2025\target3\g\high\keep"  # Folder containing .fit images
save_directory = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\Star Coords\extiction\sept 30\g"  # Where to save star coordinates
save_filename = "sept30-g-area95-star9"  # Output filename
#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════#

def open_fits(path):
    if not os.path.isfile(path):
        raise ValueError(f"Provided path is not a file: {path}")
    with fits.open(path) as fitsfile:
        return fitsfile[0].data

def zscale_plot_with_magnifier(image_data, plot_title, colorbar_title):
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(image_data)
    norm = ImageNormalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(image_data, origin="lower", norm=norm, cmap='gray_r')
    ax.set_title(plot_title)
    cbar = fig.colorbar(im)
    cbar.set_label(colorbar_title)

    zoom_factor = 5
    axins_size = 2
    axins = zoomed_inset_axes(ax, zoom=zoom_factor, loc='upper right', borderpad=1)
    axins.imshow(image_data, origin="lower", norm=norm, cmap='viridis')
    axins.set_xlim(0, 1)
    axins.set_ylim(0, 1)
    axins.axis('off')

    rect_size = 75
    rect = Rectangle((0, 0), rect_size, rect_size, edgecolor='red', facecolor='none', linewidth=1)
    ax.add_patch(rect)

    def on_mouse_move(event):
        if event.inaxes == ax:
            xdata, ydata = event.xdata, event.ydata
            if xdata is not None and ydata is not None:
                x, y = int(xdata), int(ydata)
                size = rect_size // 2
                x1 = max(x - size, 0)
                x2 = min(x + size, image_data.shape[1])
                y1 = max(y - size, 0)
                y2 = min(y + size, image_data.shape[0])

                axins.set_xlim(x1, x2)
                axins.set_ylim(y1, y2)
                axins.figure.canvas.draw_idle()

                rect.set_xy((x1, y1))
                rect.set_width(x2 - x1)
                rect.set_height(y2 - y1)
                rect.figure.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
    return fig, ax

#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════#
# Main logic
image_files = []
coordinates = []

# Collect only .fit files from the folder
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path) and filename.lower().endswith('.fit'):
        image_files.append(file_path)

image_files.sort()

def onclick(event):
    x, y = event.xdata, event.ydata
    if x is not None and y is not None:
        coords = [y, x]
        coordinates.append(coords)
        plt.close()
    else:
        print("Click inside the image area.")

for idx, file_path in enumerate(image_files):
    data = open_fits(file_path)
    plot_title = f'Image {idx+1}/{len(image_files)}: Click on reference star'
    colorbar_title = 'Pixel Intensity'
    fig, ax = zscale_plot_with_magnifier(data, plot_title, colorbar_title)
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

coordinates = np.array(coordinates)
print("Coordinates of selected points:")
print(coordinates)

# Ensure save directory exists
os.makedirs(save_directory, exist_ok=True)

# Save coordinates
save_path = os.path.join(save_directory, save_filename)
np.save(save_path, coordinates)
print(f"Coordinates array saved to {save_path}")