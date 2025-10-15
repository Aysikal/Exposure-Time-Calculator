# extinction_pipeline_fixed.py
import os
import re
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

# ---------------- CONFIG ----------------
ALIGNED_FOLDER = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\reduced\Sept 30\Area 95\g\high\keep\aligned"
OUTPUT_TABLE_FOLDER = os.path.join(ALIGNED_FOLDER, "star_tables_fixed")
os.makedirs(OUTPUT_TABLE_FOLDER, exist_ok=True)

# Hardcode RA/DEC of the target field (for airmass)
RA_HARD = "03:53:21"
DEC_HARD = "-00:00:20"

# Instrument parameters
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

SITE_TZ_NAME = "Asia/Tehran"

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

# ---------------- Helpers ----------------
def list_fits(folder):
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(('.fits', '.fit'))])

def load_fits(path):
    with fits.open(path, memmap=True) as hdul:
        data = hdul[0].data.astype(float)
        hdr = hdul[0].header
    return np.nan_to_num(data), hdr

# --- FIX: Robust DATE-OBS parser ---
def parse_dateobs_token(hdr):
    raw = hdr.get('DATE-OBS')
    if not raw:
        return None
    s = str(raw).strip()

    # Remove junk after timestamp (e.g. "NOGPS")
    s = re.sub(r'[^0-9T:\-\.]', '', s)

    # Normalize to YYYY-MM-DDTHH:MM:SS(.sss)
    match = re.match(r'(\d{4}-\d{2}-\d{2})T?(\d{2}:\d{2}:\d{2}(\.\d+)?)?', s)
    if not match:
        print(f"⚠️ Could not parse DATE-OBS: {raw}")
        return None
    if match.group(2):
        return f"{match.group(1)}T{match.group(2)}"
    else:
        return f"{match.group(1)}T00:00:00"

def local_token_to_utc_components(token, tz_name=SITE_TZ_NAME):
    if token is None:
        return None, None, None, None
    try:
        dt_local_naive = datetime.strptime(token, "%Y-%m-%dT%H:%M:%S.%f")
    except Exception:
        try:
            dt_local_naive = datetime.strptime(token, "%Y-%m-%dT%H:%M:%S")
        except Exception:
            return None, None, None, None
    try:
        tz = pytz.timezone(tz_name)
    except Exception:
        tz = pytz.FixedOffset(3 * 60 + 30)
    dt_local = tz.localize(dt_local_naive)
    dt_utc = dt_local.astimezone(pytz.utc)
    date_utc_str = dt_utc.strftime("%Y-%m-%d")
    hour_utc = dt_utc.hour
    minute_utc = dt_utc.minute
    return date_utc_str, hour_utc, minute_utc, dt_utc

def get_exptime_raw_from_header(hdr):
    for key in ("DURATION", "EXPTIME", "EXPOSURE", "EXPTIM", "ITIME", "ONTIME", "TELAPSE"):
        v = hdr.get(key)
        if v is not None:
            try:
                return float(v)
            except Exception:
                continue
    return None

def exptime_seconds_from_raw(raw_val):
    if raw_val is None:
        return np.nan
    try:
        return float(raw_val) * 1e-5
    except Exception:
        return np.nan

def centroid_in_array(arr):
    arr = np.nan_to_num(arr)
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
    if not np.any(mask):
        return np.array([])
    tbin = np.bincount(r_int[mask].ravel(), data[mask].ravel())
    nr = np.bincount(r_int[mask].ravel())
    radialprofile = np.zeros_like(tbin, dtype=float)
    valid = nr > 0
    radialprofile[valid] = tbin[valid] / nr[valid]
    radial_sm = gaussian_filter1d(radialprofile, sigma=sigma)
    return radial_sm

def interp_fwhm_from_profile(profile, n_tail=TAIL_MEDIAN):
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
    fwhm = right - left
    return float(fwhm)

def make_masks(shape, center, r_ap, r_ann_in, r_ann_out):
    yy, xx = np.indices(shape)
    rr = np.sqrt((xx - center[0])**2 + (yy - center[1])**2)
    ap_mask = rr <= r_ap
    ann_mask = (rr >= r_ann_in) & (rr <= r_ann_out)
    return ap_mask, ann_mask

# ---------------- Interactive star selection ----------------
def pick_star(image):
    plt.imshow(image, origin='lower', cmap='gray')
    plt.title("Click your reference star, then close window")
    coords = plt.ginput(1, timeout=-1)
    plt.close()
    return coords[0]

# ---------------- MAIN ----------------
fits_files = list_fits(ALIGNED_FOLDER)
if not fits_files:
    raise SystemExit("No FITS files found.")

ref_data, ref_hdr = load_fits(os.path.join(ALIGNED_FOLDER, fits_files[0]))
STAR_COORDS = pick_star(ref_data)
print(f"⭐ Using reference star at {STAR_COORDS}")

# ---------------- Loop through frames ----------------
records = []
for idx, fname in enumerate(fits_files):
    fpath = os.path.join(ALIGNED_FOLDER, fname)
    data, hdr = load_fits(fpath)

    token = parse_dateobs_token(hdr)
    date_utc_str, hour_utc, minute_utc, dt_utc = local_token_to_utc_components(token, SITE_TZ_NAME)

    from ancillary_functions import airmass_function
    try:
        airmass_val = airmass_function(date_utc_str, hour_utc, minute_utc, RA_HARD, DEC_HARD, plot_night_curve=False)
    except Exception:
        airmass_val = np.nan

    exptime_raw = get_exptime_raw_from_header(hdr)
    exptime_s = exptime_seconds_from_raw(exptime_raw)

    tiny_cut = Cutout2D(data, STAR_COORDS, REFINE_BOX, mode='partial')
    cx_local, cy_local = centroid_in_array(tiny_cut.data)
    x_star, y_star = tiny_cut.to_original_position((cx_local, cy_local))

    disp_cut = Cutout2D(data, (x_star, y_star), box_size_px, mode='partial')
    disp_data = np.nan_to_num(disp_cut.data)
    cx_disp, cy_disp = disp_cut.to_cutout_position((x_star, y_star))

    radial_profile = calculate_radial_profile(disp_data, (cx_disp, cy_disp), int(min(disp_data.shape)//2))
    fwhm_px = interp_fwhm_from_profile(radial_profile)
    if np.isnan(fwhm_px) or fwhm_px <= 0:
        fwhm_px = PSF_PIX_REF

    ap_radius = K_AP * fwhm_px
    ann_in = ANN_IN_FACT * fwhm_px
    ann_out = ANN_OUT_FACT * fwhm_px

    ap_mask, ann_mask = make_masks(disp_data.shape, (cx_disp, cy_disp), ap_radius, ann_in, ann_out)
    ap_values = disp_data[ap_mask]
    ann_values = disp_data[ann_mask]

    sum_circle = float(np.nansum(ap_values))
    n_pix = int(np.count_nonzero(ap_mask))
    median_ann = float(np.median(ann_values)) if ann_values.size > 0 else 0
    net = sum_circle - (n_pix * median_ann)
    net_per_sec = net / exptime_s if (not np.isnan(exptime_s) and exptime_s > 0) else np.nan
    minstr = -2.5 * np.log10(net_per_sec) if (not np.isnan(net_per_sec) and net_per_sec > 0) else np.nan

    records.append({
        "frame_index": idx + 1,
        "frame_name": fname,
        "DATE_OBS_token": token,
        "exptime_s": exptime_s,
        "airmass": airmass_val,
        "minstr": minstr,
        "x_star": x_star,
        "y_star": y_star
    })

    print(f"[{idx+1:02d}/{len(fits_files)}] {fname} | Airmass={airmass_val:.3f} | m_inst={minstr:.3f} | Time={token}")

# ---------------- Fit extinction ----------------
airmass_arr = np.array([r["airmass"] for r in records])
minstr_arr = np.array([r["minstr"] for r in records])
good = np.isfinite(airmass_arr) & np.isfinite(minstr_arr)
if good.sum() < 2:
    raise SystemExit("Not enough valid points for extinction fit.")

X = airmass_arr[good]
y = minstr_arr[good]
p, cov = np.polyfit(X, y, 1, cov=True)
k, m0 = p
perr = np.sqrt(np.diag(cov))
k_err, m0_err = perr

print("\n" + "="*80)
print(f"Extinction fit results:")
print(f"  k = {k:.4f} ± {k_err:.4f} mag/airmass")
print(f"  m0 = {m0:.4f} ± {m0_err:.4f} mag")
print(f"  Points used: {good.sum()}")
print("="*80)

plt.figure(figsize=(7,5))
plt.scatter(X, y, color='C0', label='data')
xs = np.linspace(np.min(X)-0.05, np.max(X)+0.05, 200)
ys = m0 + k * xs
plt.plot(xs, ys, 'r-', label=f"fit: m = m0 + kX\nk={k:.3f}±{k_err:.3f}")
plt.gca().invert_yaxis()
plt.xlabel("Airmass")
plt.ylabel("Instrumental Magnitude")
plt.title("Extinction Fit")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
