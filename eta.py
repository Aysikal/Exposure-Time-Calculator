# ======================================================
# Aperture + Annulus Photometry: Filter vs Clear Master Comparison (EXPTIME-normalized)
# ======================================================

import os
import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval, ImageNormalize
import matplotlib.pyplot as plt
import pandas as pd

# ------------------------------------------------------
# CONFIGURATION (edit these two paths only)
# ------------------------------------------------------
CLEAR_FOLDER = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\reduced and aligned\sept 30 area 95 clear HIGH 2 min"
FILTER_FOLDER = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\reduced and aligned\sept 30 area 95 g HIGH 2 min"
FILTER_NAME = "g"   # <-- label for the colored filter

# Photometry parameters
PSF_ARCSEC = 0.7
PIXEL_SCALE = 0.047 * 1.8  # arcsec/pix
AP_DIAM_ARCSEC = 2.35 * PSF_ARCSEC
AP_RADIUS_PIX = (AP_DIAM_ARCSEC / PIXEL_SCALE) / 2.0
ANN_INNER = 2.4 * AP_RADIUS_PIX
ANN_OUTER = 3.0 * AP_RADIUS_PIX

# Exposure time keys (case-insensitive)
EXPTIME_KEYS = ["EXPTIME", "EXPOSURE", "EXPTM", "EXPTIM", "TELAPSE"]

# ------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------
def list_fits(folder):
    return sorted([os.path.join(folder, f) for f in os.listdir(folder)
                   if f.lower().endswith((".fit", ".fits"))])

def load_fits(path):
    """Return (data, header) from a FITS file, ensuring numeric data."""
    with fits.open(path, memmap=True) as hdul:
        data = hdul[0].data.astype(float)
        hdr = hdul[0].header
    return np.nan_to_num(data), hdr

def get_exptime(header):
    """Try multiple exposure time keys, return float value."""
    for key in EXPTIME_KEYS:
        if key in header:
            try:
                return float(header[key])
            except Exception:
                pass
    raise KeyError(f"Exposure time not found in header keys: {EXPTIME_KEYS}")

def stack_median_exptime_normalized(fits_list):
    """Stack all FITS normalized by exposure time (counts/s)."""
    stack = []
    for f in fits_list:
        data, hdr = load_fits(f)
        exptime = get_exptime(hdr)
        if exptime <= 0:
            raise ValueError(f"Invalid exposure time ({exptime}) in {f}")
        stack.append(data / exptime)
    if not stack:
        raise ValueError("No FITS files found in folder.")
    stack = np.array(stack)
    med = np.median(stack, axis=0)
    return med

def show_and_click(image, title, nstars=None):
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(image)
    norm = ImageNormalize(vmin=vmin, vmax=vmax)
    fig, ax = plt.subplots(figsize=(9,7))
    ax.imshow(image, origin='lower', cmap='gray_r', norm=norm)
    ax.set_title(title + "\nClick on stars (press ENTER when done)")
    coords = []

    def onclick(event):
        if event.inaxes:
            coords.append((float(event.ydata), float(event.xdata)))
            ax.plot(event.xdata, event.ydata, 'o', color='lime', markersize=8)
            fig.canvas.draw()

    def onkey(event):
        if event.key in ('enter', 'return'):
            plt.close(fig)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    kid = fig.canvas.mpl_connect('key_press_event', onkey)
    plt.show()

    if nstars and len(coords) != nstars:
        raise ValueError(f"Expected {nstars} stars, got {len(coords)}.")

    return coords

def circ_sum(image, center, r):
    y0, x0 = center
    y, x = np.ogrid[:image.shape[0], :image.shape[1]]
    mask = (x-x0)**2 + (y-y0)**2 <= r**2
    return float(np.sum(image[mask])), int(np.count_nonzero(mask))

def ann_sum(image, center, r_in, r_out):
    y0, x0 = center
    y, x = np.ogrid[:image.shape[0], :image.shape[1]]
    r2 = (x-x0)**2 + (y-y0)**2
    mask = (r2 > r_in**2) & (r2 <= r_out**2)
    return float(np.sum(image[mask])), int(np.count_nonzero(mask))

# ------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------
def main():
    print("=== Loading and stacking FITS files (normalized by exposure time) ===")
    clear_files = list_fits(CLEAR_FOLDER)
    filt_files  = list_fits(FILTER_FOLDER)

    print(f"Found {len(clear_files)} clear files, {len(filt_files)} {FILTER_NAME}-filter files.")
    clear_master = stack_median_exptime_normalized(clear_files)
    filt_master  = stack_median_exptime_normalized(filt_files)

    # Save masters
    fits.writeto(os.path.join(CLEAR_FOLDER, "master_clear_counts_per_s.fits"), clear_master, overwrite=True)
    fits.writeto(os.path.join(FILTER_FOLDER, f"master_{FILTER_NAME}_counts_per_s.fits"), filt_master, overwrite=True)
    print("Master frames saved (fluxes in counts/s).")

    # === Select stars ===
    clear_coords = show_and_click(clear_master, "CLEAR MASTER — select stars")
    nstars = len(clear_coords)
    print(f"{nstars} stars selected on CLEAR master.")

    filt_coords = show_and_click(filt_master, f"{FILTER_NAME.upper()} MASTER — click same {nstars} stars in same order", nstars=nstars)

    # === Photometry per star ===
    table = []
    for idx, (pt_clear, pt_filt) in enumerate(zip(clear_coords, filt_coords), start=1):
        # Clear
        ap_sum_c, ap_n_c = circ_sum(clear_master, pt_clear, AP_RADIUS_PIX)
        ann_sum_c, ann_n_c = ann_sum(clear_master, pt_clear, ANN_INNER, ANN_OUTER)
        bkg_c = ann_sum_c / ann_n_c
        ap_flux_c = ap_sum_c - bkg_c * ap_n_c

        # Filter
        ap_sum_f, ap_n_f = circ_sum(filt_master, pt_filt, AP_RADIUS_PIX)
        ann_sum_f, ann_n_f = ann_sum(filt_master, pt_filt, ANN_INNER, ANN_OUTER)
        bkg_f = ann_sum_f / ann_n_f
        ap_flux_f = ap_sum_f - bkg_f * ap_n_f

        ratio = ap_flux_f / ap_flux_c if ap_flux_c != 0 else np.nan

        table.append({
            "Star": idx,
            "Clear_ApFlux_per_s": ap_flux_c,
            f"{FILTER_NAME}_ApFlux_per_s": ap_flux_f,
            f"{FILTER_NAME}/Clear": ratio
        })

    df = pd.DataFrame(table)
    print("\n=== Photometry Ratios (normalized by exposure time) ===")
    print(df.to_string(index=False))

    out_csv = os.path.join(FILTER_FOLDER, f"photometry_ratio_{FILTER_NAME}_norm.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved results to: {out_csv}")

# ------------------------------------------------------
if __name__ == "__main__":
    main()
