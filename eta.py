# Aperture + annulus photometry with ratios relative to clear image
import os
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import ImageNormalize, ZScaleInterval
import pandas as pd  # for table formatting

# === File paths ===
file_paths = {
    "clear": r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\reduced\Sept 30\Area 95\clear\high\target3_clear_T10C_2025_10_01_2x2_exp00.02.00.000_000001_High_6_dark_and_flat_corrected.fit",
    "g":     r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\reduced\Sept 30\Area 95\g\high\target3_g_T10C_2025_09_30_2x2_exp00.02.00.000_000001_High_1_dark_and_flat_corrected.fit",
    "r":     r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\reduced\Sept 30\Area 95\r\high\target3_r_T10C_2025_10_01_2x2_exp00.01.00.000_000001_High_5_dark_and_flat_corrected.fit",
    "i":     r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\reduced\Sept 30\Area 95\i\high\target3_i_T10C_2025_10_01_2x2_exp00.01.00.000_000001_High_5_dark_and_flat_corrected.fit",
}

# === Photometry parameters ===
psf_arcsec = 0.7
pixel_scale = 0.047 * 1.8
ap_diam_arcsec = 2.35 * psf_arcsec
ap_radius_pix = (ap_diam_arcsec / pixel_scale) / 2.0
ann_in = 2.4 * ap_radius_pix
ann_out = 3.0 * ap_radius_pix

EXPTIME_KEYS = ["EXPTIME", "EXPOSURE", "EXPTM", "EXPTIM", "TELAPSE"]

def read_image_and_header(path):
    with fits.open(path) as hdul:
        return hdul[0].data.astype(float), hdul[0].header

def get_exptime(hdr):
    for k in EXPTIME_KEYS:
        if k in hdr:
            try:
                return float(hdr[k])
            except: pass
    raise KeyError("Exposure time not found")

def show_and_click(image, title):
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(image)
    norm = ImageNormalize(vmin=vmin, vmax=vmax)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.imshow(image, origin="lower", cmap="gray_r", norm=norm)
    ax.set_title(title)
    coords = {"pt": None}
    def onclick(event):
        if event.inaxes == ax and event.xdata and event.ydata:
            coords["pt"] = (float(event.ydata), float(event.xdata))
            plt.close(fig)
    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()
    return coords["pt"]

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

def main(paths):
    results = {}
    for filt, path in paths.items():
        img, hdr = read_image_and_header(path)
        pt = show_and_click(img, f"{filt} — click reference point")
        exptime = get_exptime(hdr)
        ap_sum, ap_n = circ_sum(img, pt, ap_radius_pix)
        ann_sum_val, ann_n = ann_sum(img, pt, ann_in, ann_out)
        results[filt] = {
            "coords": pt,
            "ap_flux": ap_sum/exptime,
            "ann_flux": ann_sum_val/exptime,
            "exptime": exptime
        }

    # Ratios relative to clear
    clear_ap = results["clear"]["ap_flux"]
    clear_ann = results["clear"]["ann_flux"]

    table = []
    for filt in ["g","r","i"]:
        ap_ratio = results[filt]["ap_flux"]/clear_ap if clear_ap else None
        ann_ratio = results[filt]["ann_flux"]/clear_ann if clear_ann else None
        table.append({
            "Filter": filt,
            "Aperture_flux/s": results[filt]["ap_flux"],
            "Annulus_flux/s": results[filt]["ann_flux"],
            "Aperture/clear": ap_ratio,
            "Annulus/clear": ann_ratio
        })

    df = pd.DataFrame(table)
    print("\n=== Photometry Ratios relative to clear ===")
    print(df.to_string(index=False))
    return df

if __name__ == "__main__":
    main(file_paths)

#____________________________________________________
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.visualization import ZScaleInterval

# Optional photutils centroid
try:
    from photutils.centroids import centroid_com, centroid_2dg
    PHOTUTILS = True
except Exception:
    PHOTUTILS = False

# ---------------- CONFIG ----------------
ALIGNED_FOLDER = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\reduced\Sept 30\Area 95\g\high\keep\aligned"
OUTPUT_CUTOUT_FOLDER = os.path.join(ALIGNED_FOLDER, "star_cutouts")
os.makedirs(OUTPUT_CUTOUT_FOLDER, exist_ok=True)

PSF_ARCSEC = 0.7
PIXEL_SCALE = 0.047 * 1.8
BOX_FACTOR = 10.0        # larger cutouts so annulus fits
REFINE_RADIUS_FACTOR = 10.0  # search radius ~10*PSF for centroiding
DISPLAY_COLS = 8
Z = ZScaleInterval()

USE_GAUSSIAN = False   # <--- set True if you want centroid_2dg, else COM only

# ---------------- Derived parameters ----------------
pixels_per_arcsec = PIXEL_SCALE
box_size_px = round((BOX_FACTOR * PSF_ARCSEC) / pixels_per_arcsec)
box_size_px = box_size_px if box_size_px % 2 == 1 else box_size_px + 1

PSF_PIX = PSF_ARCSEC / PIXEL_SCALE
AP_RADIUS = 2.35 * PSF_PIX / 2.0
ANN_INNER = 2.4 * AP_RADIUS
ANN_OUTER = 3.0 * AP_RADIUS
REFINE_BOX = int(round(REFINE_RADIUS_FACTOR * PSF_PIX))
REFINE_BOX = REFINE_BOX if REFINE_BOX % 2 == 1 else REFINE_BOX + 1

print(f"Cutout box size: {box_size_px} px")
print(f"Aperture radius: {AP_RADIUS:.2f} px")
print(f"Annulus: inner={ANN_INNER:.2f} px, outer={ANN_OUTER:.2f} px")
print(f"Refinement box: {REFINE_BOX} px")

# ---------------- Helpers ----------------
def list_fits(folder):
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(('.fits', '.fit'))])

def load_fits(path):
    with fits.open(path, memmap=True) as hdul:
        data = hdul[0].data.astype(float)
        hdr = hdul[0].header
    return np.nan_to_num(data), hdr

def centroid_in_array(arr):
    arr = np.nan_to_num(arr)
    if PHOTUTILS and USE_GAUSSIAN:
        try:
            cy, cx = centroid_2dg(arr)   # Gaussian fit
            return float(cx), float(cy)
        except Exception:
            pass
    # Default: center of mass
    try:
        cy, cx = centroid_com(arr)
        return float(cx), float(cy)
    except Exception:
        total = arr.sum()
        if total <= 0:
            i, j = np.unravel_index(np.argmax(arr), arr.shape)
            return float(j), float(i)
        yy, xx = np.indices(arr.shape)
        cx = (xx * arr).sum() / total
        cy = (yy * arr).sum() / total
        return float(cx), float(cy)

# ---------------- Load reference and click stars ----------------
fits_files = list_fits(ALIGNED_FOLDER)
if not fits_files:
    raise SystemExit("No FITS files found in the aligned folder.")

ref_data, ref_hdr = load_fits(os.path.join(ALIGNED_FOLDER, fits_files[0]))

clicked = []
fig, ax = plt.subplots(figsize=(10, 8))
vmin, vmax = Z.get_limits(ref_data)
ax.imshow(ref_data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
ax.set_title(f"Click stars in {fits_files[0]}. Press Enter when done.")
def onclick(event):
    if event.inaxes:
        clicked.append((event.xdata, event.ydata))
        ax.plot(event.xdata, event.ydata, 'o', color='lime', markersize=8)
        fig.canvas.draw()
def onkey(event):
    if event.key in ('enter', 'return'):
        fig.canvas.mpl_disconnect(cid)
        fig.canvas.mpl_disconnect(kid)
        plt.close(fig)
cid = fig.canvas.mpl_connect('button_press_event', onclick)
kid = fig.canvas.mpl_connect('key_press_event', onkey)
plt.show()

if not clicked:
    raise SystemExit("No stars selected.")

# ---------------- Refine centroids on reference ----------------
refined = []
for x, y in clicked:
    try:
        cut = Cutout2D(ref_data, (x, y), REFINE_BOX, mode='partial')
        cx, cy = centroid_in_array(cut.data)
        x0 = int(round(x - cut.data.shape[1] / 2.0))
        y0 = int(round(y - cut.data.shape[0] / 2.0))
        refined.append((x0 + cx, y0 + cy))
    except Exception:
        refined.append((x, y))

# ---------------- Display overlays with anchored per-frame centroid ----------------
for sid, (x_ref, y_ref) in enumerate(refined):
    n = len(fits_files)
    cols = DISPLAY_COLS
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    axes = np.atleast_2d(axes)

    for idx, fname in enumerate(fits_files):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        data, _ = load_fits(os.path.join(ALIGNED_FOLDER, fname))

        # Local cutout around reference position for per-frame centroid
        cut = Cutout2D(data, (x_ref, y_ref), REFINE_BOX, mode='partial')
        cx_local, cy_local = centroid_in_array(cut.data)

        # Convert refined local centroid to original image coordinates
        x_star, y_star = cut.to_original_position((cx_local, cy_local))

        # Larger display cutout centered on the refined star position
        disp_cut = Cutout2D(data, (x_star, y_star), box_size_px, mode='partial')
        disp_data = disp_cut.data
        vmin, vmax = Z.get_limits(disp_data)
        ax.imshow(disp_data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)

        # Refined centroid in display cutout coordinates
        cx_disp, cy_disp = disp_cut.to_cutout_position((x_star, y_star))

        # Draw aperture and annulus centered on the refined centroid
        ap = Circle((cx_disp, cy_disp), AP_RADIUS, edgecolor='red', facecolor='none', lw=1.5)
        ann1 = Circle((cx_disp, cy_disp), ANN_INNER, edgecolor='yellow', facecolor='none', lw=1.0, linestyle='dashed')
        ann2 = Circle((cx_disp, cy_disp), ANN_OUTER, edgecolor='yellow', facecolor='none', lw=1.0, linestyle='dashed')
        ax.add_patch(ann2)
        ax.add_patch(ann1)
        ax.add_patch(ap)

        ax.set_title(f"{idx+1}/{n}", fontsize=8)
        ax.axis('off')

    # Hide any empty subplots
    for idx in range(n, rows*cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis('off')

    plt.suptitle(f"star_{sid+1} photometry overlays (per-frame COM centroid)", fontsize=14)
    plt.tight_layout()
    plt.show()
