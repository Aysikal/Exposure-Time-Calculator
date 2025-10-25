import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit

# ---------------- CONFIG ----------------
ALIGNED_FOLDER = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\data 2\Rezaei_Altafi_10_07_2025\standard_star\94 B2\i\high\keep"
COORDS_PATH = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\Star Coords\standard_star\94 B2\oct7-i-94B2.npy"
PSF_ARCSEC = 1
PIXEL_SCALE = 0.047 * 1.8
BOX_FACTOR = 10.0
PSF_PIX_REF = PSF_ARCSEC / PIXEL_SCALE
box_size_px = round((BOX_FACTOR * PSF_ARCSEC) / PIXEL_SCALE)
if box_size_px % 2 == 0:
    box_size_px += 1

# --- Overexposure detection threshold (12-bit CCD) ---
CCD_LIMIT = 4095
OVEREXPOSURE_THRESHOLD = 4080

# ---------------- Helpers ----------------
def list_fits(folder):
    """Return all FITS/FIT files in a folder, sorted alphabetically."""
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(('.fits', '.fit'))])

def load_fits(path):
    """Load a FITS image and return its data as a float array."""
    with fits.open(path, memmap=False) as hdul:
        data = np.nan_to_num(hdul[0].data.astype(float))
    return data

def gaussian_2d(coord, amplitude, x0, y0, sx, sy, offset):
    """2D Gaussian model for PSF fitting."""
    x, y = coord
    return amplitude * np.exp(-0.5 * (((x - x0) / sx) ** 2 + ((y - y0) / sy) ** 2)) + offset

# ---------------- Main logic ----------------
fits_files = list_fits(ALIGNED_FOLDER)
if not fits_files:
    raise SystemExit("No FITS files found in folder.")

coords = np.load(COORDS_PATH)
if coords.ndim != 2 or coords.shape[1] < 2:
    raise SystemExit("Coordinate file must have at least 2 columns: [orig_y, orig_x]")

# Add columns for refined coords if missing
if coords.shape[1] == 2:
    coords = np.hstack([coords, np.full((coords.shape[0], 2), np.nan)])

refined_coords = []
cutouts = []
fits_models = []
refined_positions = []
overexposure_flags = []

for idx, fname in enumerate(fits_files):
    file_path = os.path.join(ALIGNED_FOLDER, fname)
    data = load_fits(file_path)

    orig_y, orig_x, refined_y, refined_x = coords[idx][:4]


    if not (np.isfinite(refined_x) and np.isfinite(refined_y)):
        refined_y, refined_x = orig_y, orig_x

    x_star, y_star = float(refined_x), float(refined_y)
    half_box = box_size_px // 2
    x_c = int(round(x_star))
    y_c = int(round(y_star))

    x1 = max(0, x_c - half_box)
    x2 = min(data.shape[1], x_c + half_box + 1)
    y1 = max(0, y_c - half_box)
    y2 = min(data.shape[0], y_c + half_box + 1)

    subimg = data[y1:y2, x1:x2].astype(float)
    if subimg.size == 0:
        print(f"⚠ Empty cutout for {fname}; skipping")
        refined_coords.append([orig_y, orig_x, np.nan, np.nan])
        overexposure_flags.append(False)
        continue

    # --- Overexposure check ---
    max_val = np.nanmax(subimg)
    is_overexposed = max_val >= OVEREXPOSURE_THRESHOLD
    if is_overexposed:
        print(f"⚠ Star overexposed in {fname}: max={max_val:.0f} ADU (CCD limit = {CCD_LIMIT})")

    yy, xx = np.mgrid[y1:y2, x1:x2]

    amp_guess = np.nanmax(subimg) - np.nanmedian(subimg)
    if amp_guess <= 0:
        amp_guess = np.nanmax(subimg)
    init_x0 = x_star
    init_y0 = y_star
    init_sx = max(1.0, PSF_PIX_REF)
    init_sy = max(1.0, PSF_PIX_REF)
    init_off = np.nanmedian(subimg)

    init_guess = (amp_guess, init_x0, init_y0, init_sx, init_sy, init_off)

    try:
        popt, _ = curve_fit(
            lambda coord_flat, amplitude, x0, y0, sx, sy, offset:
                gaussian_2d((xx, yy), amplitude, x0, y0, sx, sy, offset).ravel(),
            xdata=None,
            ydata=subimg.ravel(),
            p0=init_guess,
            maxfev=5000
        )
        amp_fit, x_fit, y_fit, sx_fit, sy_fit, off_fit = popt

        model = gaussian_2d((xx, yy), *popt)
        refined_coords.append([orig_y, orig_x, float(y_fit), float(x_fit)])
        cutouts.append(subimg)
        fits_models.append(model)
        refined_positions.append((x_fit - x1, y_fit - y1))
        overexposure_flags.append(is_overexposed)

        print(f"Frame {idx+1}/{len(fits_files)} refined center: ({x_fit:.2f}, {y_fit:.2f})")

    except Exception as e:
        print(f"⚠ Fit failed for frame {idx+1} ({fname}): {e}")
        refined_coords.append([orig_y, orig_x, np.nan, np.nan])
        overexposure_flags.append(is_overexposed)

# Save results
refined_coords = np.hstack([np.array(refined_coords), np.array(overexposure_flags).reshape(-1, 1)])
np.save(COORDS_PATH, refined_coords)

print(f"\n✅ Saved refined coordinates to {COORDS_PATH}")
print("Columns: [orig_y, orig_x, refined_y, refined_x, overexposed_flag]")
