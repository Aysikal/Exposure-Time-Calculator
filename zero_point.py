import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.nddata import Cutout2D
from scipy.optimize import curve_fit
from matplotlib.patches import Circle

# === User config ===
fits_path = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\Sep30\Rezaei_30_sep_2025\target3\i\high\keep\hot pixels removed\aligned\aligned_target3_i_T10C_2025_10_01_2x2_exp00.01.00.000_000001_High_3_cycleclean_iter3.fit"
catalog_path = r"C:\Users\AYSAN\Desktop\project\INO\ETC\PAN-star\area95_stars_compact.csv"

star_IDs = [42, 30, 43, 74, 75, 76, 63, 48, 79, 65, 61, 28, 89, 2, 4, 70, 54, 18, 62, 82, 80, 13, 12]
star_positions_g = [
    (793, 893), (809, 1113), (807, 753), (1170, 509), (1159, 381), (1159, 328),
    (1583, 604), (1785, 581), (1409, 409), (1599, 409), (1466, 1063),
    (1710, 1033), (908, 1448), (1171, 1704), (1561, 1632), (589, 1423),
    (336, 1292), (1127, 1276), (1674, 704), (1963, 657), (1906, 880),
    (195, 1418), (173, 1468)
]
star_positions = [(x-11, y+46) for x, y in star_positions_g]

cutout_half = 25
REFINE_BOX = 25
gain = 1 / 16.5
readnoise = 3.7
inner_radius_factor = 2.5
outer_radius_factor = 3.0

# === Helper functions ===
def circular_mask(shape, center, radius):
    yy, xx = np.indices(shape)
    cx, cy = center
    return (xx - cx)**2 + (yy - cy)**2 <= radius**2

def annulus_mask(shape, center, r_in, r_out):
    yy, xx = np.indices(shape)
    cx, cy = center
    r2 = (xx - cx)**2 + (yy - cy)**2
    return (r2 >= r_in**2) & (r2 <= r_out**2)

def gaussian_2d(xy, amp, x0, y0, sigma_x, sigma_y, theta, offset):
    x, y = xy
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    return offset + amp * np.exp(-(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2))

# === Load FITS ===
data = fits.getdata(fits_path)
hdr = fits.getheader(fits_path)
ny, nx = data.shape

# Exposure time
exptime_s = None
for key in ("EXPTIME", "EXPOSURE", "TOTAL_EXP", "NETEXPT"):
    if key in hdr:
        try:
            exptime_s = float(hdr[key]) * 10**(-5)
            break
        except Exception:
            pass
if exptime_s is None:
    raise ValueError("Exposure time not found in header")

# Load catalog
catalog = pd.read_csv(catalog_path)
def get_known_mag(star_id):
    row = catalog[catalog['ID'] == star_id]
    if len(row) == 0:
        print(f"ID {star_id} not found in catalog")
        return np.nan
    return float(row['i_sdss'].values[0])

results = []
refined_positions = []

# === Loop through stars ===
for star_id, (x_star, y_star) in zip(star_IDs, star_positions):
    known_r_mag = get_known_mag(star_id)
    if np.isnan(known_r_mag):
        continue

    try:
        tiny = Cutout2D(data, (x_star, y_star), (REFINE_BOX, REFINE_BOX), mode='partial')
    except Exception as e:
        print(f"Skipping star {star_id} at ({x_star},{y_star}): {e}")
        continue

    y, x = np.indices(tiny.data.shape)
    x_flat = x.ravel()
    y_flat = y.ravel()
    data_flat = tiny.data.ravel()

    amp_guess = np.nanmax(tiny.data)
    x0_guess, y0_guess = REFINE_BOX/2, REFINE_BOX/2
    sigma_guess = 2.0
    theta_guess = 0.0
    offset_guess = np.nanmedian(tiny.data)
    p0 = [amp_guess, x0_guess, y0_guess, sigma_guess, sigma_guess, theta_guess, offset_guess]

    try:
        popt, _ = curve_fit(gaussian_2d, (x_flat, y_flat), data_flat, p0=p0, maxfev=3000)
        amp_fit, x0_fit, y0_fit, sigma_x_fit, sigma_y_fit, theta_fit, offset_fit = popt
        if 0 <= x0_fit <= REFINE_BOX and 0 <= y0_fit <= REFINE_BOX:
            x_refined = tiny.position_original[0] - REFINE_BOX/2 + x0_fit
            y_refined = tiny.position_original[1] - REFINE_BOX/2 + y0_fit
        else:
            x_refined, y_refined = tiny.position_original
    except:
        x_refined, y_refined = tiny.position_original

    refined_positions.append((x_refined, y_refined))

    try:
        disp_cut = Cutout2D(data, (x_refined, y_refined), (2*cutout_half, 2*cutout_half), mode='partial')
    except Exception as e:
        print(f"Skipping star {star_id} at refined ({x_refined},{y_refined}): {e}")
        continue

    disp = np.nan_to_num(disp_cut.data)
    cx_disp, cy_disp = cutout_half, cutout_half

    radii = np.arange(2, 13.1, 0.5)
    snrs = []
    for r in radii:
        ap_mask = circular_mask(disp.shape, (cx_disp, cy_disp), r)
        ann_mask = annulus_mask(disp.shape, (cx_disp, cy_disp), r*inner_radius_factor, r*outer_radius_factor)
        n_pix = np.count_nonzero(ap_mask)
        sum_star = np.nansum(disp[ap_mask])
        mean_bg = np.nanmedian(disp[ann_mask]) if np.any(ann_mask) else np.nanmedian(disp)
        net_counts = sum_star - n_pix*mean_bg
        S_e = net_counts * gain
        sky_e = n_pix * mean_bg * gain
        var_e = max(S_e, 0) + sky_e + n_pix*(readnoise**2)
        noise_e = np.sqrt(max(var_e, 1e-9))
        snr = S_e / noise_e if noise_e > 0 else 0
        snrs.append(snr)

    snrs = np.array(snrs)
    best_idx = np.argmax(snrs)
    best_r = radii[best_idx]
    max_snr = snrs[best_idx]

    ap_mask = circular_mask(disp.shape, (cx_disp, cy_disp), best_r)
    ann_mask = annulus_mask(disp.shape, (cx_disp, cy_disp), best_r*inner_radius_factor, best_r*outer_radius_factor)
    sum_star = np.nansum(disp[ap_mask])
    n_pix = np.count_nonzero(ap_mask)
    mean_bg = np.nanmedian(disp[ann_mask]) if np.any(ann_mask) else np.nanmedian(disp)
    net_counts = sum_star - n_pix * mean_bg
    net_per_sec = net_counts / exptime_s if exptime_s > 0 else np.nan
    m_instr = -2.5 * np.log10(net_per_sec) if net_per_sec > 0 else np.nan
    ZP = known_r_mag + 2.5 * np.log10(net_per_sec) if net_per_sec > 0 else np.nan

    results.append({
        'ID': star_id,
        'x_init': x_star,
        'y_init': y_star,
        'x_refined': x_refined,
        'y_refined': y_refined,
        'best_radius': best_r,
        'max_snr': max_snr,
        'instr_mag': m_instr,
        'known_mag': known_r_mag,
        'ZP': ZP
    })

    # SNR plot
    plt.figure(figsize=(5,4))
    plt.plot(radii, snrs, '-o', label=f'Star {star_id}')
    plt.axvline(best_r, color='r', ls='--', label='Best radius')
    plt.xlabel("Aperture radius (pixels)")
    plt.ylabel("SNR")
    plt.title(f"SNR vs radius — Star {star_id}")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Save results
df = pd.DataFrame(results)
df.to_csv("photometry_results_i.csv", index=False)
print("Results saved to photometry_results_i.csv")

# ZP statistics
ZP_values = df['ZP'].dropna().values
mean_ZP = np.mean(ZP_values)
median_ZP = np.median(ZP_values)
std_ZP = np.std(ZP_values, ddof=1)
error_ZP = std_ZP / np.sqrt(len(ZP_values))
approx_error = (max(ZP_values)-min(ZP_values))/6

print("\n--- ZP Statistics ---")
print(f"Mean     = {mean_ZP:.4f}")
print(f"Median   = {median_ZP:.4f}")
print(f"Std Dev  = {std_ZP:.4f}")
print(f"Error    = {error_ZP:.4f}")
print(f"Approx Err (Range/6) = {approx_error:.4f}")

# ZP histogram
plt.figure(figsize=(6,4))
plt.hist(ZP_values, bins=10, color='lightblue', edgecolor='k')
plt.axvline(mean_ZP, color='r', linestyle='--', label=f'Mean ZP = {mean_ZP:.3f}')
plt.xlabel("Zero Point (ZP)")
plt.ylabel("Count")
plt.title("Distribution of Zero Points (r-filter)")
plt.legend()
plt.tight_layout()
plt.show()

# === Show stars on image with apertures ===
plt.figure(figsize=(10,10))
plt.imshow(data, origin='lower', cmap='gray', vmin=np.percentile(data,5), vmax=np.percentile(data,99))
plt.title("Stars with Refined Positions and Apertures")

for i, row in df.iterrows():
    # Refined center
    x, y = row['x_refined'], row['y_refined']
    best_r = row['best_radius']
    plt.plot(row['x_init'], row['y_init'], 'ro', markersize=5, alpha=0.5)  # initial guess
    plt.plot(x, y, 'gx', markersize=8, label='Refined' if i==0 else "")
    # Aperture circle
    circ = Circle((x, y), best_r, edgecolor='yellow', facecolor='none', lw=1)
    plt.gca().add_patch(circ)

plt.xlabel("X Pixel")
plt.ylabel("Y Pixel")
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
