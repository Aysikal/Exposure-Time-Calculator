# extinction_pipeline.py
import os
import math
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pytz

from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.visualization import ZScaleInterval
from scipy.ndimage import gaussian_filter1d

# import your airmass routine (must be on PYTHONPATH)
from ancillary_functions import airmass_function

# ---------------- CONFIG ----------------
ALIGNED_FOLDER = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\reduced\Sept 30\Area 95\g\high\keep\aligned"
OUTPUT_TABLE_FOLDER = os.path.join(ALIGNED_FOLDER, "star_tables")
os.makedirs(OUTPUT_TABLE_FOLDER, exist_ok=True)

# Hardcode RA/DEC of the target field used for airmass calculation (strings)
RA_HARD = "03:53:21"
DEC_HARD = "-00:00:20"

# Instrument / geometric parameters
PSF_ARCSEC = 0.7
PIXEL_SCALE = 0.047 * 1.8
BOX_FACTOR = 10.0
REFINE_RADIUS_FACTOR = 10.0
Z = ZScaleInterval()
USE_GAUSSIAN = False

# Aperture recipe
K_AP = 0.9
ANN_IN_FACT = 3.0
ANN_OUT_FACT = 5.0
RADIAL_SMOOTH_SIGMA = 2.0
TAIL_MEDIAN = 15

# Site timezone name for parsing DATE-OBS tokens (your DATE-OBS is local Asia/Tehran)
SITE_TZ_NAME = "Asia/Tehran"

# Display / plotting options
PLOT_SAVEFIG = False
PLOT_FILENAME = os.path.join(OUTPUT_TABLE_FOLDER, "extinction_fit.png")

# ---------------- Derived ----------------
pixels_per_arcsec = PIXEL_SCALE
box_size_px = round((BOX_FACTOR * PSF_ARCSEC) / pixels_per_arcsec)
if box_size_px % 2 == 0:
    box_size_px += 1

PSF_PIX_REF = PSF_ARCSEC / PIXEL_SCALE
REFINE_BOX = int(round(REFINE_RADIUS_FACTOR * PSF_PIX_REF))
if REFINE_BOX % 2 == 0:
    REFINE_BOX += 1

print(f"box_size_px = {box_size_px}, REFINE_BOX = {REFINE_BOX}, PSF_PIX_REF = {PSF_PIX_REF:.3f}")

# ---------------- Helpers (photometry + headers parsing) ----------------
def list_fits(folder):
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(('.fits', '.fit'))])

def load_fits(path):
    with fits.open(path, memmap=True) as hdul:
        data = hdul[0].data.astype(float)
        hdr = hdul[0].header
    return np.nan_to_num(data), hdr

def get_exptime_raw_from_header(hdr):
    for key in ("DURATION", "EXPTIME", "EXPOSURE", "EXPTIM", "ITIME", "ONTIME", "TELAPSE"):
        v = hdr.get(key)
        if v is not None:
            try:
                return float(v)
            except Exception:
                try:
                    return float(str(v))
                except Exception:
                    continue
    return None

def exptime_seconds_from_raw(raw_val):
    # raw_val stored as counts of 10 microseconds (10 us = 1e-5 s)
    if raw_val is None:
        return np.nan
    try:
        return float(raw_val) * 1e-5
    except Exception:
        return np.nan

def parse_dateobs_token(hdr):
    s = hdr.get('DATE-OBS')
    if s is None:
        return None
    s = str(s).strip()
    # Remove any trailing tokens like 'NOGPS'
    s = s.split()[0]
    # Ensure we preserve fractional seconds if present
    if "T" in s:
        date_part, time_part = s.split("T", 1)
        # Keep only the numeric time part (e.g., '01:59:00.785')
        time_part = ''.join(ch for ch in time_part if ch.isdigit() or ch in [":", "."])
        token = f"{date_part}T{time_part}"
    else:
        token = s
    return token

def local_token_to_utc_components(token, tz_name=SITE_TZ_NAME):
    if token is None:
        return None, None, None, None
    # Accept tokens like 'YYYY-MM-DDTHH:MM:SS' with optional fractional seconds
    try:
        if "T" in token:
            date_part, time_part = token.split("T", 1)
        elif " " in token:
            date_part, time_part = token.split(" ", 1)
        else:
            return None, None, None, None
        dt = None
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
            try:
                dt_local_naive = datetime.strptime(f"{date_part}T{time_part}", fmt)
                dt = dt_local_naive
                break
            except Exception:
                continue
        if dt is None:
            tp = time_part.split(".")[0]
            dt = datetime.strptime(f"{date_part}T{tp}", "%Y-%m-%dT%H:%M:%S")
    except Exception:
        return None, None, None, None
    try:
        tz = pytz.timezone(tz_name)
    except Exception:
        tz = pytz.FixedOffset(3 * 60 + 30)
    try:
        dt_local = tz.localize(dt)
    except Exception:
        if dt.tzinfo is None:
            dt_local = dt.replace(tzinfo=tz)
        else:
            dt_local = dt
    dt_utc = dt_local.astimezone(pytz.utc)
    date_utc_str = dt_utc.strftime("%Y-%m-%d")
    hour_utc = dt_utc.hour
    minute_utc = dt_utc.minute
    return date_utc_str, hour_utc, minute_utc, dt_utc

# simple centroid by COM; photutils used optionally inside if installed
def centroid_in_array(arr):
    arr = np.nan_to_num(arr)
    if arr.size == 0:
        return 0.0, 0.0
    try:
        # photutils centroid_com available? use it if installed
        from photutils.centroids import centroid_com, centroid_2dg
        cy, cx = centroid_com(arr)
        return float(cx), float(cy)
    except Exception:
        total = arr.sum()
        if total <= 0:
            i, j = np.unravel_index(np.nanargmax(arr), arr.shape)
            return float(j), float(i)
        yy, xx = np.indices(arr.shape)
        cx = (xx * arr).sum() / total
        cy = (yy * arr).sum() / total
        return float(cx), float(cy)

def calculate_radial_profile(data, center, max_radius, sigma=RADIAL_SMOOTH_SIGMA):
    y, x = np.indices(data.shape)
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r_int = r.astype(int)
    mask = r_int <= max_radius
    vals = np.nan_to_num(data)
    if not np.any(mask):
        return np.array([])
    tbin = np.bincount(r_int[mask].ravel(), vals[mask].ravel())
    nr = np.bincount(r_int[mask].ravel())
    radialprofile = np.zeros_like(tbin, dtype=float)
    valid = nr > 0
    radialprofile[valid] = tbin[valid] / nr[valid]
    radial_sm = gaussian_filter1d(radialprofile, sigma=sigma)
    return radial_sm

def interp_fwhm_from_profile(profile, n_tail=TAIL_MEDIAN):
    profile = np.asarray(profile, dtype=float)
    if profile.size < 3:
        return np.nan
    tail = profile[-n_tail:] if profile.size > n_tail else profile[int(profile.size/2):]
    baseline = np.median(tail)
    peak = np.max(profile)
    half = (peak + baseline) / 2.0
    radii = np.arange(profile.size)
    mask = profile >= half
    if not mask.any():
        return np.nan
    left = np.argmax(mask)
    right = len(mask) - 1 - np.argmax(mask[::-1])
    if left == 0:
        left_cross = 0.0
    else:
        x0, x1 = radii[left-1], radii[left]
        y0, y1 = profile[left-1], profile[left]
        left_cross = float(x0 if y1 == y0 else x0 + (half - y0) * (x1 - x0) / (y1 - y0))
    if right == radii[-1]:
        right_cross = float(radii[-1])
    else:
        x0, x1 = radii[right], radii[right+1]
        y0, y1 = profile[right], profile[right+1]
        right_cross = float(x0 if y1 == y0 else x0 + (half - y0) * (x1 - x0) / (y1 - y0))
    fwhm = right_cross - left_cross
    return float(fwhm)

def make_masks(shape, center, r_ap, r_ann_in, r_ann_out):
    yy, xx = np.indices(shape)
    rr = np.sqrt((xx - center[0])**2 + (yy - center[1])**2)
    ap_mask = rr <= r_ap
    ann_mask = (rr >= r_ann_in) & (rr <= r_ann_out)
    return ap_mask, ann_mask

# ---------------- Interactive selection on reference image ----------------
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

fits_files = list_fits(ALIGNED_FOLDER)
if not fits_files:
    raise SystemExit("No FITS files found in ALIGNED_FOLDER.")

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

# ---------------- Refine centroids on reference image ----------------
refined = []
for x, y in clicked:
    try:
        cut = Cutout2D(ref_data, (x, y), REFINE_BOX, mode='partial')
        cx, cy = centroid_in_array(cut.data)
        x0 = (x - cut.data.shape[1] / 2.0)
        y0 = (y - cut.data.shape[0] / 2.0)
        refined.append((x0 + cx, y0 + cy))
    except Exception:
        refined.append((x, y))

# ---------------- Per-star photometry and airmass collection ----------------
# We'll collect arrays across all frames for a given star, fit minstr vs airmass, and plot.
for sid, (x_ref, y_ref) in enumerate(refined):
    records = []
    for idx, fname in enumerate(fits_files):
        data, hdr = load_fits(os.path.join(ALIGNED_FOLDER, fname))

        # parse local DATE-OBS token and convert to UTC components for airmass_function
        token = parse_dateobs_token(hdr)
        date_utc_str, hour_utc, minute_utc, dt_utc = local_token_to_utc_components(token, tz_name=SITE_TZ_NAME)

        # compute airmass via your ancillary function
        if date_utc_str is None:
            airmass_val = np.nan
        else:
            try:
                airmass_val = airmass_function(date_utc_str, hour_utc, minute_utc, RA_HARD, DEC_HARD, plot_night_curve=False)
            except Exception:
                airmass_val = np.nan

        # exposure
        exptime_raw = get_exptime_raw_from_header(hdr)
        exptime_s = exptime_seconds_from_raw(exptime_raw)

        # per-frame centroid refine
        tiny_cut = Cutout2D(data, (x_ref, y_ref), REFINE_BOX, mode='partial')
        cx_local, cy_local = centroid_in_array(tiny_cut.data)
        dx = cx_local - tiny_cut.data.shape[1] / 2.0
        dy = cy_local - tiny_cut.data.shape[0] / 2.0
        if math.hypot(dx, dy) > REFINE_BOX / 2.0:
            cx_local = tiny_cut.data.shape[1] / 2.0
            cy_local = tiny_cut.data.shape[0] / 2.0
        x_star, y_star = tiny_cut.to_original_position((cx_local, cy_local))

        # display cutout and masks
        disp_cut = Cutout2D(data, (x_star, y_star), box_size_px, mode='partial')
        disp_data = np.nan_to_num(disp_cut.data)
        cx_disp, cy_disp = disp_cut.to_cutout_position((x_star, y_star))

        max_radius = int(min(disp_data.shape) // 2) - 1
        if max_radius < 3:
            fwhm_px = max(PSF_PIX_REF, 1.0)
        else:
            radial_profile = calculate_radial_profile(disp_data, (cx_disp, cy_disp), max_radius)
            fwhm_meas = interp_fwhm_from_profile(radial_profile)
            fwhm_px = fwhm_meas if (not np.isnan(fwhm_meas) and fwhm_meas > 0) else max(PSF_PIX_REF, 1.0)

        ap_radius = K_AP * fwhm_px
        ann_in = ANN_IN_FACT * fwhm_px
        ann_out = ANN_OUT_FACT * fwhm_px

        ap_mask, ann_mask = make_masks(disp_data.shape, (cx_disp, cy_disp), ap_radius, ann_in, ann_out)
        ap_values = disp_data[ap_mask]
        ann_values = disp_data[ann_mask]

        sum_circle = float(np.nansum(ap_values))
        n_pix = int(np.count_nonzero(ap_mask))
        if ann_values.size == 0:
            median_ann = float(np.median(disp_data.ravel()))
        else:
            median_ann = float(np.median(ann_values))
        net = sum_circle - (n_pix * median_ann)
        net_per_sec = net / exptime_s if (not np.isnan(exptime_s) and exptime_s > 0) else np.nan

        # instrument magnitude: only valid when net_per_sec > 0
        if (not np.isnan(net_per_sec)) and (net_per_sec > 0):
            minstr = -2.5 * np.log10(net_per_sec)
        else:
            minstr = np.nan

        # local date/time strings for CSV
        date_local = None
        time_local = None
        if token is not None:
            parts = token.split("T") if "T" in token else token.split(" ")
            if len(parts) == 2:
                date_local, time_local = parts[0], parts[1]
            else:
                date_local = token
                time_local = None

        records.append({
            "frame_index": idx + 1,
            "frame_name": fname,
            "date_local": date_local,
            "time_local": time_local,
            "exptime_raw_10us": exptime_raw,
            "exptime_s": exptime_s,
            "airmass": airmass_val,
            "x_refined": float(x_star),
            "y_refined": float(y_star),
            "sum_circle": sum_circle,
            "n_pix": n_pix,
            "median_annulus": median_ann,
            "bg_subtracted_sum": net,
            "bg_subtracted_per_sec": net_per_sec,
            "minstr": minstr
        })

    # Save CSV for this star
    out_csv = os.path.join(OUTPUT_TABLE_FOLDER, f"star_{sid+1:02d}_extinction_input.csv")
    fieldnames = [
        "frame_index", "frame_name", "date_local", "time_local",
        "exptime_raw_10us", "exptime_s", "airmass",
        "x_refined", "y_refined",
        "sum_circle", "n_pix", "median_annulus", "bg_subtracted_sum", "bg_subtracted_per_sec", "minstr"
    ]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rec in records:
            w.writerow(rec)

    # Build arrays for fitting: keep only rows with finite airmass and minstr
    airmass_arr = np.array([r["airmass"] for r in records], dtype=float)
    minstr_arr = np.array([r["minstr"] for r in records], dtype=float)

    good = np.isfinite(airmass_arr) & np.isfinite(minstr_arr)
    if good.sum() < 2:
        print(f"Star {sid+1}: not enough valid points to fit extinction (found {good.sum()}) — skipping.")
        continue

    X = airmass_arr[good]
    y = minstr_arr[good]

    # linear fit y = m0 + k * X  -> slope k is extinction
    # Use np.polyfit to get covariance
    p, cov = np.polyfit(X, y, 1, cov=True)
    k = p[0]      # slope
    m0 = p[1]     # intercept
    # Extract uncertainties from covariance matrix
    perr = np.sqrt(np.diag(cov))
    k_err = perr[0]
    m0_err = perr[1]

    print("\n" + "="*80)
    print(f"Star {sid+1} extinction fit results:")
    print(f"  Extinction k = {k:.4f} ± {k_err:.4f} mag / airmass")
    print(f"  Instrumental zero point m0 = {m0:.4f} ± {m0_err:.4f} mag")
    print(f"  Number of points used = {good.sum()}")
    print("="*80)

    # Plot minstr vs airmass and fitted line
    plt.figure(figsize=(7,5))
    plt.scatter(X, y, color='C0', label='data', zorder=5)
    # fitted line
    xs = np.linspace(np.min(X)-0.05, np.max(X)+0.05, 200)
    ys = m0 + k * xs
    plt.plot(xs, ys, color='C1', lw=2, label=f"fit: m = m0 + k X\nk={k:.4f}±{k_err:.4f}")
    plt.gca().invert_yaxis()  # magnitudes: smaller (brighter) up
    plt.xlabel("Airmass")
    plt.ylabel("Instrumental magnitude (mag)")
    plt.title(f"Star {sid+1} instrumental mag vs airmass")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if PLOT_SAVEFIG:
        plt.savefig(os.path.join(OUTPUT_TABLE_FOLDER, f"star_{sid+1:02d}_extinction_plot.png"), dpi=200)
    plt.show()
