import os
import shutil
import tkinter as tk
from tkinter import filedialog
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- CONFIG ---
zscale = ZScaleInterval()

# --- GUI Setup ---
root = tk.Tk()
root.title("Keep or Trash FITS Viewer")

# Set a large default window size (not fullscreen)
root.geometry("1200x800")  # Width x Height in pixels
root.resizable(True, True)  # Allow resizing

# --- File Handling ---
folder_path = filedialog.askdirectory(title="Select Folder with FITS Files")
fits_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.fits', '.fit'))]
current_index = 0

# --- Subfolders ---
keep_folder = os.path.join(folder_path, "keep")
trash_folder = os.path.join(folder_path, "trash")
os.makedirs(keep_folder, exist_ok=True)
os.makedirs(trash_folder, exist_ok=True)

# --- Display Area ---
fig, ax = plt.subplots(figsize=(8, 6))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

def show_image():
    ax.clear()
    if current_index >= len(fits_files):
        ax.text(0.5, 0.5, "All files reviewed!", ha='center', va='center', fontsize=24)
        canvas.draw()

        # Disable Keep and Trash buttons
        for widget in btn_frame.winfo_children():
            widget.config(state=tk.DISABLED)

        # Add Exit button
        tk.Button(btn_frame, text="Exit", command=root.destroy,
                  font=('Helvetica', 20), height=2, width=20).pack(pady=20)
        return

    file_path = os.path.join(folder_path, fits_files[current_index])
    try:
        with fits.open(file_path) as hdul:
            data = hdul[0].data
            if data is not None:
                vmin, vmax = zscale.get_limits(data)
                ax.imshow(data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
                ax.set_title(fits_files[current_index], fontsize=16)
            else:
                ax.text(0.5, 0.5, "No image data", ha='center', va='center', fontsize=16)
    except Exception as e:
        ax.text(0.5, 0.5, f"Error loading file:\n{e}", ha='center', va='center', fontsize=14)

    canvas.draw()


def move_file(destination):
    global current_index
    if current_index < len(fits_files):
        src = os.path.join(folder_path, fits_files[current_index])
        dst = os.path.join(destination, fits_files[current_index])
        shutil.move(src, dst)
        current_index += 1
        show_image()

# --- Buttons ---
btn_frame = tk.Frame(root)
btn_frame.pack(side=tk.BOTTOM, pady=30)

button_style = {'font': ('Helvetica', 20), 'height': 2, 'width': 20}

tk.Button(btn_frame, text="✅ Keep", command=lambda: move_file(keep_folder), **button_style).pack(side=tk.LEFT, padx=40)
tk.Button(btn_frame, text="🗑️ Trash", command=lambda: move_file(trash_folder), **button_style).pack(side=tk.RIGHT, padx=40)

# --- Start ---
show_image()
root.mainloop()
