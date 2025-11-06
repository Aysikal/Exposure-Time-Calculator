import os
import math
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.visualization import ZScaleInterval, ImageNormalize

# === User config ===
fits_path = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\rezaei_saba_farideH_2025_10_22\GRB251013c\high\hot pixels removed\aligned\reduced\stacked_sum.fits"
targets = {
    "GRB_251013C": (2090.9, 1497.5),
    "Ref_star": (2372.0, 2110.0)
}
cutout_half = 100
REFINE_BOX = 25


# Known reference magnitude (if you still want calibration later)
ref_known_mag = 20.06

# Default instrument params (fallbacks)
DEFAULT_GAIN = 1/16.5   # e-/ADU
DEFAULT_READNOISE = 5.0  # e-

# SNR scan parameters
radius_step = 0.5
inner_radius_factor = 2.5
outer_radius_factor = 3.0

# --- PSF / pixel scale constraints (you provided pixel scale = 0.047 * 1.8) ---
pixel_scale_arcsec = 0.047 * 1.8   # arcsec / pixel -> 0.0846 arcsec/pixel
min_fwhm_arcsec = 1.0              # your note: PSF at least 1 arcsecond

# compute minimum FWHM/HWHM in pixels from those physical constraints
min_fwhm_pixels = float(min_fwhm_arcsec / pixel_scale_arcsec)
min_hwhm_pixels = min_fwhm_pixels / 2.0

# --- helper masks ---
def circular_mask(shape, center, radius):
    yy, xx = np.indices(shape)
    cx, cy = center
    r2 = (xx - cx)**2 + (yy - cy)**2
    return r2 <= (radius**2)

def annulus_mask(shape, center, r_in, r_out):
    yy, xx = np.indices(shape)
    cx, cy = center
    r2 = (xx - cx)**2 + (yy - cy)**2
    return (r2 >= r_in**2) & (r2 <= r_out**2)

# --- simple centroid ---
def centroid_in_array(arr):
    y, x = np.indices(arr.shape)
    arr_pos = np.where(arr > 0, arr, 0.0)
    total = arr_pos.sum()
    if total <= 0:
        return arr.shape[1]/2.0, arr.shape[0]/2.0
    cx = (x * arr_pos).sum() / total
    cy = (y * arr_pos).sum() / total
    return cx, cy

# --- estimate HWHM from radial profile (approx) ---
def estimate_hwhm(cutout, center):
    img = cutout.astype(float)
    yy, xx = np.indices(img.shape)
    r = np.sqrt((xx - center[0])**2 + (yy - center[1])**2)
    r_flat = r.ravel()
    img_flat = img.ravel()
    r_max = max(4.0, min(img.shape)/2.0)
    nbins = int(np.ceil(r_max*4))
    bins = np.linspace(0.0, r_max, nbins+1)
    idx = np.digitize(r_flat, bins) - 1
    # use median in radial bins to be robust to outliers
    radial_med = np.array([np.median(img_flat[idx==i]) if np.any(idx==i) else 0.0 for i in range(len(bins)-1)])
    radii = 0.5*(bins[:-1] + bins[1:])
    peak = radial_med.max()
    if peak <= 0:
        return 1.0
    half = peak / 2.0
    try:
        j = np.where(radial_med <= half)[0][0]
        hwhm = radii[j]
        if hwhm <= 0:
            return 1.0
        return float(hwhm)
    except IndexError:
        return float(radii[-1])

# --- improved SNR vs radius finder (uses FWHM-aware search range) ---
def get_radius(image, center_xy, HWHM, gain, readnoise,
               radius_step=0.5,
               inner_radius=inner_radius_factor, outer_radius=outer_radius_factor):
    """
    center_xy: (x, y) pixel coordinates
    Returns: best_radius, max_snr, snrs, radii_list
    """
    cx = float(center_xy[0])
    cy = float(center_xy[1])
    yy, xx = np.indices(image.shape)
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    image = image.astype(np.float64)

    # ensure HWHM is sane: use estimate but enforce minimum from pixel scale / user expectation
    if not np.isfinite(HWHM) or HWHM <= 0:
        HWHM = 1.0
    HWHM = max(HWHM, min_hwhm_pixels)  # enforce minimum HWHM in pixels based on PSF size
    FWHM = max(2.0 * HWHM, min_fwhm_pixels)  # final FWHM (px) used for search guidance

    # radius search range around expected PSF size (in pixels)
    # allow radii from ~0.6*FWHM up to ~3.5*FWHM (but never exceed cutout boundary)
    radius_min = max(1.0, 0.6 * FWHM)
    max_possible = max(1.0, min(image.shape) / 2.0 - 1.0)
    radius_max = min(max_possible, max(radius_min + radius_step, 3.5 * FWHM, 8.0))

    if radius_max <= radius_min:
        radius_max = radius_min + radius_step * 4.0

    best_radius = radius_min
    max_snr = -np.inf
    snrs, radii_list = [], []

    for radius in np.arange(radius_min, radius_max + radius_step/2.0, radius_step):
        star_mask = dist <= radius
        n_star_pix = float(np.count_nonzero(star_mask))
        if n_star_pix <= 0:
            snrs.append(0.0)
            radii_list.append(radius)
            continue

        sum_brightness = float(np.nansum(image[star_mask]))

        ann_mask = (dist > inner_radius * radius) & (dist <= outer_radius * radius)
        ann_pixels = image[ann_mask]
        if ann_pixels.size == 0:
            mean_bg = float(np.nanmedian(image))
        else:
            mean_bg = float(np.nanmedian(ann_pixels))

        background_brightness = mean_bg * n_star_pix
        net_counts = sum_brightness - background_brightness

        # convert to electrons and compute variance correctly
        S_e = net_counts * gain
        sky_e = n_star_pix * mean_bg * gain
        photon_term = max(S_e, 0.0)               # only positive contribution to photon variance from star
        var_e = photon_term + sky_e + n_star_pix * (readnoise**2)
        noise_e = np.sqrt(max(var_e, 1e-9))
        snr = S_e / noise_e if noise_e > 0 else 0.0

        snrs.append(float(snr))
        radii_list.append(float(radius))
        if snr > max_snr:
            max_snr = float(snr)
            best_radius = float(radius)

    if max_snr == -np.inf:
        max_snr = 0.0
        best_radius = radius_min

    return best_radius, max_snr, snrs, radii_list

# === Load image and header ===
data = fits.getdata(fits_path)
hdr = fits.getheader(fits_path)

# get total exposure time if present, else set to 2 hours
exptime_s = None
for key in ("EXPTIME", "EXPOSURE", "TOTAL_EXP", "NETEXPT"):
    if key in hdr:
        try:
            exptime_s = float(hdr[key])
            break
        except Exception:
            pass
if exptime_s is None:
    exptime_s = 6739
print(f"Using total exposure time (s) = {exptime_s}")

# instrument params
gain = 1/16.5
readnoise = 3.7
print(f"Using gain={gain} e-/ADU, readnoise={readnoise} e-")
print(f"Pixel scale = {pixel_scale_arcsec:.6f} arcsec/px -> min FWHM enforced = {min_fwhm_pixels:.2f} px (={min_fwhm_arcsec} arcsec)")

# quick full-image preview with targets
zscale = ZScaleInterval()
norm_full = ImageNormalize(data, interval=zscale)
plt.figure(figsize=(8,8))
plt.imshow(data, origin='lower', cmap='gray_r', norm=norm_full)
for name,(x,y) in targets.items():
    plt.plot(x, y, marker='+', color='yellow', ms=10)
    plt.text(x+8, y+8, name, color='yellow', fontsize=9)
plt.title(os.path.basename(fits_path))
plt.colorbar(label='ADU')
plt.tight_layout()
plt.show()

results = {}

for name, (x_ref, y_ref) in targets.items():
    print(f"\nProcessing {name} at approx ({x_ref:.1f}, {y_ref:.1f})")

    try:
        tiny = Cutout2D(data, (x_ref, y_ref), (REFINE_BOX, REFINE_BOX), mode='partial')
        cx_local, cy_local = centroid_in_array(tiny.data)
        x_star, y_star = tiny.to_original_position((cx_local, cy_local))
        dx, dy = cx_local - tiny.data.shape[1]/2, cy_local - tiny.data.shape[0]/2
        if math.hypot(dx, dy) > REFINE_BOX/2:
            x_star, y_star = x_ref, y_ref
    except Exception:
        x_star, y_star = x_ref, y_ref

    disp_cut = Cutout2D(data, (x_star, y_star), (2*cutout_half, 2*cutout_half), mode='partial')
    disp = np.nan_to_num(disp_cut.data)
    cx_disp, cy_disp = disp_cut.to_cutout_position((x_star, y_star))

    HWHM = estimate_hwhm(disp, (cx_disp, cy_disp))
    # apply physical minimum HWHM from pixel scale
    HWHM_used = max(HWHM, min_hwhm_pixels)
    FWHM_used = max(2.0 * HWHM_used, min_fwhm_pixels)
    print(f"  estimated HWHM (raw) = {HWHM:.2f} px, using HWHM = {HWHM_used:.2f} px (FWHM ~ {FWHM_used:.2f} px)")

    best_r, max_snr, snrs, radii = get_radius(
        disp, (cx_disp, cy_disp), HWHM_used,
        gain=gain, readnoise=readnoise,
        radius_step=radius_step,
        inner_radius=inner_radius_factor,
        outer_radius=outer_radius_factor
    )
    print(f"  best radius = {best_r:.2f} px, max SNR = {max_snr:.2f}")

    ap_mask = circular_mask(disp.shape, (cx_disp, cy_disp), best_r)
    ann_mask = annulus_mask(disp.shape, (cx_disp, cy_disp), inner_radius_factor*best_r, outer_radius_factor*best_r)

    ap_vals = disp[ap_mask]
    ann_vals = disp[ann_mask]
    sum_circle = float(np.nansum(ap_vals)) if ap_vals.size>0 else 0.0
    n_pix = int(np.count_nonzero(ap_mask))
    mean_ann = float(np.nanmedian(ann_vals)) if ann_vals.size>0 else float(np.nanmedian(disp))
    net = sum_circle - n_pix * mean_ann
    net_per_sec = net / exptime_s if (exptime_s and exptime_s>0) else np.nan
    m_instr = -2.5 * np.log10(net_per_sec)  if (net_per_sec>0 and np.isfinite(net_per_sec)) else np.nan

    results[name] = {
        "x_star": x_star, "y_star": y_star, "HWHM": HWHM_used,
        "best_radius": best_r, "max_snr": max_snr,
        "sum_circle": sum_circle, "n_pix": n_pix,
        "mean_annulus": mean_ann, "net": net,
        "net_per_sec": net_per_sec, "instrumental_mag": m_instr,
        "snrs": snrs, "radii": radii, "disp": disp, "center": (cx_disp, cy_disp)
    }

    # --- plots ---
    plt.figure(figsize=(6,3.5))
    plt.plot(radii, snrs, '-o', ms=4)
    plt.axvline(best_r, color='red', ls='--', label=f"best r={best_r:.2f}")
    # also show a guideline at 1.3 * FWHM (approx typical)
    plt.axvline(1.3 * FWHM_used, color='green', ls=':', label=f"1.3*FWHM={1.3*FWHM_used:.2f}px")
    plt.xlabel("Aperture radius (px)")
    plt.ylabel("SNR")
    plt.title(f"SNR vs radius for {name}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(5,5))
    norm_cut = ImageNormalize(disp, interval=ZScaleInterval())
    ax.imshow(disp, origin='lower', cmap='gray_r', norm=norm_cut)
    circ_ap = plt.Circle((cx_disp, cy_disp), best_r, edgecolor='red', facecolor='none', lw=1.2)
    circ_in = plt.Circle((cx_disp, cy_disp), inner_radius_factor*best_r, edgecolor='cyan', facecolor='none', lw=0.9, ls='--')
    circ_out = plt.Circle((cx_disp, cy_disp), outer_radius_factor*best_r, edgecolor='cyan', facecolor='none', lw=0.9, ls='--')
    ax.add_patch(circ_ap); ax.add_patch(circ_in); ax.add_patch(circ_out)
    ax.plot(cx_disp, cy_disp, marker='+', color='yellow', ms=10)
    ax.set_title(f"{name} (r={best_r:.2f}px, SNR={max_snr:.2f})")
    ax.text(5, 10, f"m_instr = {m_instr:.3f}", color='white', fontsize=9, bbox=dict(facecolor='black', alpha=0.6, pad=2))
    plt.show()

# --- Differential photometry calibration (if available) ---
if "RGB_251013C" in results and "Ref_star" in results:
    F_target = results["RGB_251013C"]["net_per_sec"]
    F_ref = results["Ref_star"]["net_per_sec"]

    if F_target > 0 and F_ref > 0:
        RGB_cal_mag = ref_known_mag - 2.5 * np.log10(F_target / F_ref)
        # magnitude uncertainty from SNR (if SNR available)
        snr_target = results["RGB_251013C"]["max_snr"]
        snr_ref = results["Ref_star"]["max_snr"]
        # approximate magnitude error propagation (dominant terms)
        sigma_m_target = 1.0857 / snr_target if snr_target > 0 else np.nan
        sigma_m_ref = 1.0857 / snr_ref if snr_ref > 0 else np.nan
        # combined error (in quadrature)
        sigma_RGB_mag = np.sqrt(np.nan_to_num(sigma_m_target**2) + np.nan_to_num(sigma_m_ref**2))
    else:
        RGB_cal_mag = np.nan
        sigma_RGB_mag = np.nan

    print(f"\nCalibrated magnitude for RGB_251013C = {RGB_cal_mag:.4f} ± {sigma_RGB_mag:.3f}")
else:
    print("\nWarning: Both target and reference fluxes not found for calibration.")

# --- summary printout ---
print("\nPhotometry summary:")
for name, r in results.items():
    print(f"{name}: x={r['x_star']:.2f}, y={r['y_star']:.2f}, HWHM={r['HWHM']:.2f}, best_r={r['best_radius']:.2f}, SNR={r['max_snr']:.2f}, m_instr={r['instrumental_mag']:.3f}")
