import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import LogStretch, ImageNormalize
from scipy.optimize import curve_fit
from scipy.ndimage import center_of_mass
from photutils.aperture import CircularAperture, CircularAnnulus

# ------------------------------- Configuration -------------------------------
# Path to the .npy file containing coordinates (rows correspond to frames)
COORDS_PATH = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\Star Coords\standard_star\oct7-g-96A4.npy"
filter = "g"
# Folder containing the images you want to analyze (example: green 'g' reduced folder)
IMAGE_FOLDER = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\data 2\Rezaei_Altafi_10_07_2025\standard_star\96 A4\g\high\keep\reduced"

# Box (cutout) size around the star (pixels)
BOXSIZE = 150

# Instrument parameters (as provided)
gain = 1/16.5     # e-/ADU
readnoise = 3.6     # e-

# CCD overexposure parameters
CCD_LIMIT = 4095
OVEREXPOSURE_THRESHOLD = 4095  # set to 4000 for early warning if desired

# Aperture search settings
RADIUS_STEP = 0.5
INNER_ANNULUS_FACTOR = 2.5    # annulus inner radius = factor * aperture_radius
OUTER_ANNULUS_FACTOR = 3.0    # annulus outer radius = factor * aperture_radius

# Number of columns for cutout grid plot
N_COLS = 3

# Output CSV for summary results
OUTPUT_CSV = os.path.join(os.path.dirname(COORDS_PATH), f"snr_aperture_results_{filter}.csv")

# ------------------------------- Helper functions -------------------------------
def list_fits(folder):
    """Return sorted FITS/FIT files in a folder."""
    return sorted([os.path.join(folder, f) for f in os.listdir(folder)
                   if f.lower().endswith(('.fits', '.fit', '.fts'))])

def load_fits(path):
    """Load primary HDU data as float array (nan -> 0)."""
    with fits.open(path, memmap=False) as hdul:
        data = np.nan_to_num(hdul[0].data.astype(float))
    return data

def twoD_Gaussian(coords, amplitude, xo, yo, sigma_x, sigma_y, offset):
    """2D Gaussian used for fitting (coords: (x.ravel(), y.ravel()))."""
    x, y = coords
    g = offset + amplitude * np.exp(
        -(((x - xo) ** 2) / (2 * sigma_x ** 2) + ((y - yo) ** 2) / (2 * sigma_y ** 2))
    )
    return g.ravel()

def aperture_masks_from_center(shape, center, radius):
    """
    Create boolean mask for circle of given radius at center (y,x).
    Returns mask and number of pixels (n_pix).
    """
    yy, xx = np.indices(shape)
    dist = np.sqrt((xx - center[1])**2 + (yy - center[0])**2)
    mask = dist <= radius
    return mask, mask.sum()

def annulus_mask(shape, center, r_in, r_out):
    yy, xx = np.indices(shape)
    dist = np.sqrt((xx - center[1])**2 + (yy - center[0])**2)
    mask = (dist > r_in) & (dist <= r_out)
    return mask, mask.sum()

def get_best_radius_by_snr(image, center, HWHM, radius_step=0.5,
                           inner_factor=INNER_ANNULUS_FACTOR,
                           outer_factor=OUTER_ANNULUS_FACTOR,
                           gain=gain, readnoise=readnoise):
    """
    Search aperture radii to maximize SNR for a star cutout image.
    center: (y, x) in cutout coordinates (floats)
    HWHM: half width at half maximum in pixels (float)
    Returns: best_radius, best_snr, radii_list, snrs_list
    """
    # sensible radius search range around PSF
    radius_min = max(0.5, HWHM / 2.0)
    radius_max = max(2.0, HWHM * 2.0)

    radii = np.arange(radius_min, radius_max + radius_step, radius_step)
    snrs = []

    # precompute grid distances
    yy, xx = np.indices(image.shape)
    dist = np.sqrt((xx - center[1])**2 + (yy - center[0])**2)

    for r in radii:
        # star aperture
        star_mask = dist <= r
        n_star = star_mask.sum()
        if n_star == 0:
            snrs.append(0.0)
            continue
        sum_star = np.sum(image[star_mask])

        # background annulus
        r_in = inner_factor * r
        r_out = outer_factor * r
        ann_mask = (dist > r_in) & (dist <= r_out)
        n_ann = ann_mask.sum()
        if n_ann == 0:
            bkg_mean = np.median(image[~star_mask])  # fallback to global median excluding star
        else:
            bkg_mean = np.mean(image[ann_mask])

        # background contribution inside aperture:
        bkg_total = bkg_mean * n_star

        # Convert to electrons (gain is e-/ADU)
        signal_e = (sum_star - bkg_total) * gain
        # ensure signal_e non-negative for sqrt (but still allow small positive)
        signal_e = max(signal_e, 0.0)

        # Noise components (in electrons)
        obj_plus_bkg_e = (sum_star) * gain  # includes background contribution in aperture
        bkg_contrib_e = n_star * (bkg_mean * gain)
        noise_e = np.sqrt(obj_plus_bkg_e + (n_star * (readnoise ** 2)) + bkg_contrib_e)

        # Guard division by zero
        snr = (signal_e / noise_e) if noise_e > 0 else 0.0
        snrs.append(snr)

    snrs = np.array(snrs)
    best_idx = np.nanargmax(snrs)
    return float(radii[best_idx]), float(snrs[best_idx]), radii.tolist(), snrs.tolist()

# ------------------------------- Main processing -------------------------------
def main():
    # load coords: robust to various column counts
    coords = np.load(COORDS_PATH)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise SystemExit("Coordinate file must have at least 2 columns: [orig_y, orig_x]")

    fits_list = list_fits(IMAGE_FOLDER)
    if len(fits_list) == 0:
        raise SystemExit(f"No FITS files found in {IMAGE_FOLDER}")

    # we'll process up to the number of rows in coords or number of fits, whichever is smaller
    n_files = min(len(fits_list), coords.shape[0])
    if len(fits_list) != coords.shape[0]:
        print(f"⚠ WARNING: number of FITS files ({len(fits_list)}) != rows in coords ({coords.shape[0]}). "
              f"Processing first {n_files} frames (sorted order).")

    results = []

    # SNR figure (single)
    snr_fig, snr_ax = plt.subplots(figsize=(10, 7))

    # Create cutout grid sized to fit all processed images
    n_images = n_files
    n_cols = N_COLS
    n_rows = int(np.ceil(n_images / n_cols)) if n_images > 0 else 1
    cutout_fig, cutout_axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 5 * n_rows))
    # reshape axes array into 2D (works even when n_rows or n_cols == 1)
    cutout_axes = np.array(cutout_axes).reshape(n_rows, n_cols)

    # loop over frames we will process
    for idx in range(n_files):
        path = fits_list[idx]
        row = coords[idx]
        # always take first four columns (orig_y, orig_x, refined_y, refined_x) if present
        if row.size >= 4:
            orig_y, orig_x, refined_y, refined_x = row[:4]
        else:
            orig_y, orig_x = row[0], row[1]
            refined_y, refined_x = np.nan, np.nan

        # fallback if refined not finite
        if not (np.isfinite(refined_x) and np.isfinite(refined_y)):
            refined_y, refined_x = orig_y, orig_x

        img = load_fits(path)

        # Determine integer center for cutout; keep fractional center for photometry fitting
        cy = float(refined_y)
        cx = float(refined_x)
        y0 = int(round(cy))
        x0 = int(round(cx))

        # Define cutout boundaries
        half = BOXSIZE // 2
        y1 = max(0, y0 - half)
        y2 = min(img.shape[0], y0 + half)
        x1 = max(0, x0 - half)
        x2 = min(img.shape[1], x0 + half)
        cut = img[y1:y2, x1:x2].astype(float)
        if cut.size == 0:
            print(f"⚠ Empty cutout for {os.path.basename(path)}; skipping")
            continue

        # Overexposure check (pixel at or above threshold)
        maxval = np.nanmax(cut)
        overexposed = (maxval >= OVEREXPOSURE_THRESHOLD)
        if overexposed:
            print(f"⚠ Star overexposed in {os.path.basename(path)}: max={maxval:.0f} ADU (>= {OVEREXPOSURE_THRESHOLD})")

        # compute center-of-mass in cutout to use as initial guess for Gaussian
        try:
            com = center_of_mass(cut)
        except Exception:
            # fallback to approximate center
            com = ((cut.shape[0] - 1) / 2.0, (cut.shape[1] - 1) / 2.0)

        # fit 2D Gaussian to cutout
        y_indices, x_indices = np.indices(cut.shape)
        coords_for_fit = (x_indices.ravel(), y_indices.ravel())
        amplitude_guess = np.nanmax(cut) - np.nanmedian(cut)
        if amplitude_guess <= 0:
            amplitude_guess = np.nanmax(cut)
        xo_guess = com[1]
        yo_guess = com[0]
        sigma_guess = 3.0
        offset_guess = np.nanmedian(cut)
        p0 = (amplitude_guess, xo_guess, yo_guess, sigma_guess, sigma_guess, offset_guess)

        try:
            popt, pcov = curve_fit(twoD_Gaussian, coords_for_fit, cut.ravel(), p0=p0, maxfev=8000)
            amp_fit, xo_fit, yo_fit, sx_fit, sy_fit, off_fit = popt
        except Exception as e:
            print(f"⚠ Gaussian fit failed for {os.path.basename(path)}: {e}")
            xo_fit, yo_fit = com[1], com[0]
            sx_fit = sy_fit = sigma_guess

        # estimate HWHM: FWHM = 2.355 * sigma ; HWHM = FWHM / 2 = 1.1775 * sigma
        sigma_avg = (sx_fit + sy_fit) / 2.0
        HWHM = 1.1775 * sigma_avg

        # compute best aperture via SNR maximization using cutout-local coords
        best_r, best_snr, radii_list, snrs = get_best_radius_by_snr(
            cut, (yo_fit, xo_fit), HWHM, radius_step=RADIUS_STEP,
            inner_factor=INNER_ANNULUS_FACTOR, outer_factor=OUTER_ANNULUS_FACTOR,
            gain=gain, readnoise=readnoise
        )

        # For plotting: aperture center for photutils plotting is (x, y)
        aperture_center_cut = (xo_fit, yo_fit)

        # Plot SNR curve (infer filter by filename token)
        fname = os.path.basename(path)
        lower = fname.lower()
        if 'g' in lower and not any(k in lower for k in ('gr','g_')):  # naive heuristic
            filter_key = 'g'
        elif 'r' in lower:
            filter_key = 'r'
        elif 'i' in lower:
            filter_key = 'i'
        elif 'u' in lower:
            filter_key = 'u'
        else:
            # fallback: look for tokens in folder name
            folder_lower = os.path.basename(IMAGE_FOLDER).lower()
            filter_key = 'g' if 'g' in folder_lower else ('r' if 'r' in folder_lower else 'i')

        color_map = {'g': 'green', 'r': 'red', 'i': 'purple', 'u': 'blue'}
        color = color_map.get(filter_key, 'black')

        linestyle = '-' if (idx % 2 == 1) else '--'
        base = os.path.splitext(fname)[0]
        short_name = base if len(base) <= 30 else base[:15] + "..."
        label = f"{short_name} (SNR={best_snr:.2f}, r={best_r:.2f}px)"
        snr_ax.plot(radii_list, snrs, linestyle=linestyle, color=color, label=label)


        # Plot cutout on the corresponding axes
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        ax = cutout_axes[row_idx, col_idx]
        norm = ImageNormalize(vmin=np.nanpercentile(cut, 5), vmax=np.nanpercentile(cut, 99), stretch=LogStretch())
        ax.imshow(cut, origin='lower', norm=norm)

        circ = CircularAperture((aperture_center_cut[0], aperture_center_cut[1]), r=best_r)
        ann = CircularAnnulus((aperture_center_cut[0], aperture_center_cut[1]),
                              r_in=best_r * INNER_ANNULUS_FACTOR, r_out=best_r * OUTER_ANNULUS_FACTOR)
        circ.plot(ax=ax, color=color, lw=2)
        ann.plot(ax=ax, color='yellow', lw=1.5)
        ax.set_title(f"{fname}\nSNR={best_snr:.2f} r={best_r:.2f}px")
        ax.set_xticks([]); ax.set_yticks([])

        # store result
        results.append({
            'filename': fname,
            'orig_y': float(orig_y),
            'orig_x': float(orig_x),
            'refined_y': float(refined_y),
            'refined_x': float(refined_x),
            'best_radius_px': best_r,
            'best_snr': best_snr,
            'overexposed': bool(overexposed)
        })

        print(f"[{idx+1}/{n_files}] {fname} -> best r={best_r:.2f} px, SNR={best_snr:.2f}, overexposed={overexposed}")

    # finalize SNR plot
    snr_ax.set_xlabel("Aperture Radius (pixels)")
    snr_ax.set_ylabel("SNR")
    snr_ax.set_title("SNR vs Aperture Radius")
    snr_ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0))
    snr_ax.grid(True)
    snr_fig.tight_layout()

    # If there are empty subplots (because grid > n_images), hide their axes
    total_axes = n_rows * n_cols
    for extra_idx in range(n_images, total_axes):
        r = extra_idx // n_cols
        c = extra_idx % n_cols
        ax_extra = cutout_axes[r, c]
        ax_extra.axis('off')

    cutout_fig.suptitle("Star cutouts with chosen aperture and background annulus", fontsize=14)
    cutout_fig.tight_layout()

    # Save results to CSV
    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'orig_y', 'orig_x', 'refined_y', 'refined_x', 'best_radius_px', 'best_snr', 'overexposed']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"\n✅ Saved results for {len(results)} frames to: {OUTPUT_CSV}")

    # Show diagnostic plots
    plt.show()


if __name__ == "__main__":
    main()
