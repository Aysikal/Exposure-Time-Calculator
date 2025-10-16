#!/usr/bin/env python3

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ZScaleInterval

# ------------------------------------------------------------
# Hardcoded folder
# ------------------------------------------------------------
FOLDER = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\Sep30\Rezaei_30_sep_2025\target3\g\high\2 min\aligned"

# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------

def list_fits(folder):
    """Return sorted list of FITS files in a folder."""
    fits_exts = ("*.fits", "*.fit", "*.fz", "*.fts")
    files = []
    for pat in fits_exts:
        files.extend(sorted(glob.glob(os.path.join(folder, pat))))
    return files


def read_primary(path):
    """
    Read FITS primary image as float array.
    If the data has multiple planes, always use plane 0.
    """
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data
        if data is None:
            raise ValueError(f"No image data in {path}")

        data = np.asarray(data, dtype=float)

        # Handle 3D cube (e.g. (2, H, W))
        if data.ndim == 3:
            print(f"ℹ️ {os.path.basename(path)} → using plane 0 from {data.shape}")
            data = data[0]

        if data.ndim != 2:
            raise ValueError(f"Unexpected shape {data.shape} in {path}")

        return data


def compute_display_limits(img):
    """Compute display stretch using zscale, fallback to percentiles."""
    try:
        zscale = ZScaleInterval()
        vmin, vmax = zscale.get_limits(img)
    except Exception:
        vmin = np.nanpercentile(img, 1)
        vmax = np.nanpercentile(img, 99.5)
    return vmin, vmax


def show_blink_sequence(folder, delay=0.3):
    """Blink through all FITS images in a folder to check alignment."""
    files = list_fits(folder)
    if len(files) == 0:
        print(f"No FITS files found in {folder}")
        return

    print(f"🧭 Found {len(files)} FITS files in: {folder}")
    imgs = [read_primary(f) for f in files]

    # Display setup
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.ion()
    vmin, vmax = compute_display_limits(imgs[0])
    im = ax.imshow(imgs[0], cmap="gray", origin="lower", vmin=vmin, vmax=vmax, interpolation="none")
    title = ax.set_title(os.path.basename(files[0]))
    plt.axis("off")

    print("✨ Blinking images — press Ctrl+C to stop ✨")

    try:
        while True:
            for f, img in zip(files, imgs):
                vmin, vmax = compute_display_limits(img)
                im.set_data(img)
                im.set_clim(vmin, vmax)
                title.set_text(os.path.basename(f))
                plt.pause(delay)
    except KeyboardInterrupt:
        print("\n👋 Stopped by user.")
        plt.ioff()
        plt.show()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    folder = FOLDER
    if not os.path.isdir(folder):
        print(f"Folder not found: {folder}")
        return
    show_blink_sequence(folder)


if __name__ == "__main__":
    main()
