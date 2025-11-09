import numpy as np
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval, ImageNormalize

# === User settings ===
fits_path = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\rezaei_saba_farideH_2025_10_22\GRB251013c\high\hot pixels removed\aligned\3 min\aligned_grb_i_2025_10_22_1x1_exp00.03.20.000_000001_High_1_cycleclean_iter3.fit"
exptime_s = 1 # seconds, integrated total exposure
print(f"Using fixed total exposure time (s) = {exptime_s}")
K_i = 0.15
K_g = 0.35
K_r = 0.2
X = 1.3
# approximate pixel coordinates (x, y)
targets = {
    "GRB_251013C": (2090.9, 1497.5),
    "Ref_star_95": (2370.0, 2110.0), 
    "Ref_star_82" : (2807, 2759),
    "Ref_star_26" : (2629, 1472),
    "Ref_star_97" : (2384, 774),
    "Ref_star_4" : (2079, 662),
    "Ref_star_21" : (713, 2548),
}

# known magnitudes of reference stars
ref_stars_mag = {
    "Ref_star_95": 19.20388806,
    "Ref_star_82": 18.32350113,
    "Ref_star_26": 19.356297,
    "Ref_star_97":17.79016791,
    "Ref_star_4": 18.05083694,
    "Ref_star_21": 19.59700745
}

aperture_radius = 8   # pixels
inner_annulus = 15    # pixels
outer_annulus = 20    # pixels

# === Read noise ===
read_noise_adu = 3.5

# === Load FITS data ===
data = fits.getdata(fits_path)
hdr = fits.getheader(fits_path)

# === Utility Functions ===
def circular_mask(shape, center, radius):
    yy, xx = np.indices(shape)
    cx, cy = center
    return (xx - cx)**2 + (yy - cy)**2 <= radius**2

def annulus_mask(shape, center, r_in, r_out):
    yy, xx = np.indices(shape)
    cx, cy = center
    r2 = (xx - cx)**2 + (yy - cy)**2
    return (r2 >= r_in**2) & (r2 <= r_out**2)

def gaussian_2d(xy, amp, x0, y0, sigma_x, sigma_y, offset):
    x, y = xy
    g = offset + amp * np.exp(-(((x - x0)**2 / (2*sigma_x**2)) + ((y - y0)**2 / (2*sigma_y**2))))
    return g.ravel()

def refine_center(img, x0, y0, box=7):
    """Fit a 2D Gaussian to refine star center."""
    x0i = int(round(x0))
    y0i = int(round(y0))
    xmin = max(x0i - box, 0)
    xmax = min(x0i + box + 1, img.shape[1])
    ymin = max(y0i - box, 0)
    ymax = min(y0i + box + 1, img.shape[0])
    
    cut = img[ymin:ymax, xmin:xmax]
    yy, xx = np.indices(cut.shape)
    
    amp0 = cut.max() - cut.min()
    offset0 = cut.min()
    sigma0 = 2.0
    p0 = [amp0, cut.shape[1]/2, cut.shape[0]/2, sigma0, sigma0, offset0]
    
    try:
        popt, _ = curve_fit(gaussian_2d, (xx, yy), cut.ravel(), p0=p0)
        refined_x = xmin + popt[1]
        refined_y = ymin + popt[2]
        return refined_x, refined_y
    except:
        return x0, y0

def measure_flux(image, x, y, r, rin, rout, exptime):
    cutout_size = 120   # or 100, 150, etc.
    cutout = Cutout2D(image, (x, y), cutout_size, mode='partial')

    cx, cy = cutout.to_cutout_position((x, y))
    img = cutout.data

    # refine center
    cx, cy = refine_center(img, cx, cy)

    ap_mask = circular_mask(img.shape, (cx, cy), r)
    ann_mask = annulus_mask(img.shape, (cx, cy), rin, rout)

    ap_values = img[ap_mask]
    ann_values = img[ann_mask]

    _, med_bg, std_bg = sigma_clipped_stats(ann_values, sigma=3.0)

    sum_ap = np.nansum(ap_values)
    n_pix = np.count_nonzero(ap_mask)
    net_flux = sum_ap - n_pix * med_bg

    # Normalize by exposure time
    normalized_flux = net_flux / exptime

    return normalized_flux, med_bg, std_bg, n_pix, img, (cx, cy)

# === Measure fluxes ===
results = {}
print("\n=== Normalized Instrumental Flux Measurements (per second) ===")
for name, (x, y) in targets.items():
    net, bg, std_bg, n, img, center = measure_flux(
        data, x, y, aperture_radius, inner_annulus, outer_annulus, exptime_s
    )
    results[name] = {"net": net, "bg": bg, "std_bg": std_bg, "n_pix": n, "center": center, "img": img}
    inst_mag = -2.5 * np.log10(net) if net > 0 else np.nan
    print(f"{name}: net_flux = {net:.4e} counts/s, background = {bg:.2f}, std_bg = {std_bg:.2f}, n_pix = {n}, instrumental mag = {inst_mag:.3f}")

# === Plot cutouts with aperture/annulus ===
for name, (x, y) in targets.items():
    cut = results[name]["img"]
    cx, cy = results[name]["center"]
    norm = ImageNormalize(cut, interval=ZScaleInterval())
    plt.figure(figsize=(5,5))
    plt.imshow(cut, origin='lower', cmap='gray_r', norm=norm)
    plt.plot(cx, cy, 'r+', ms=10)
    circ = plt.Circle((cx, cy), aperture_radius, edgecolor='red', fill=False)
    ann_in = plt.Circle((cx, cy), inner_annulus, edgecolor='cyan', fill=False, ls='--')
    ann_out = plt.Circle((cx, cy), outer_annulus, edgecolor='cyan', fill=False, ls='--')
    plt.gca().add_patch(circ)
    plt.gca().add_patch(ann_in)
    plt.gca().add_patch(ann_out)
    plt.title(name)
    plt.show()

# === Differential photometry ===
if "GRB_251013C" in results and ref_stars_mag:
    F_target = results["GRB_251013C"]["net"]
    mag_list = []

    print("\n=== Calibrated Magnitudes (per reference star) ===")
    for ref_name, ref_mag in ref_stars_mag.items():
        if ref_name in results:
            F_ref = results[ref_name]["net"]
            if F_target > 0 and F_ref > 0:
                mag = ref_mag - 2.5 * np.log10(F_target / F_ref)
                mag_list.append(mag)
                print(f"Using {ref_name}: GRB_mag = {mag:.3f} (Ref mag = {ref_mag:.3f})")
    if mag_list:
        mean_mag = np.mean(mag_list)
        std_mag = np.std(mag_list)
        print(f"\nMean calibrated magnitude for GRB_251013C = {mean_mag:.3f} ± {std_mag:.3f}")

# === Compute Zero Point ===
zp_list = []
for ref_name, ref_mag in ref_stars_mag.items():
    if ref_name in results:
        F_ref = results[ref_name]["net"]
        if F_ref > 0:
            zp = ref_mag + 2.5 * np.log10(F_ref)
            zp_list.append(zp)

ZP = np.mean(zp_list)
corrected_ZP = ZP + (K_i)*(X)
ZP_std = np.std(zp_list)
print(f"\nPhotometric Zero Point = {ZP:.3f} ± {ZP_std:.3f}")
print(f"\nPhotometric Zero Point corrected for k = {corrected_ZP:.3f}")
# === Compute limiting magnitudes with read noise included ===
# Sky sigma from reference star annulus
sigma_sky_adu = None
for rn in ref_stars_mag.keys():
    if rn in results:
        cut = results[rn]["img"]
        cx, cy = results[rn]["center"]
        ann = annulus_mask(cut.shape, (cx, cy), inner_annulus, outer_annulus)
        _, _, sigma_sky_adu = sigma_clipped_stats(cut[ann], sigma=3.0)
        break

N_pix = results["GRB_251013C"]["n_pix"]

# Total noise per pixel inside aperture
sigma_total_per_pixel = np.sqrt(sigma_sky_adu**2 + read_noise_adu**2)

# Total aperture noise
sigma_total_ap = np.sqrt(N_pix) * sigma_total_per_pixel

def limiting_mag_snr(corrected_ZP, sigma_total_ap, exptime_s, SNR):
    F_lim = SNR * sigma_total_ap / exptime_s       # counts/s
    m_lim = corrected_ZP - 2.5 * np.log10(F_lim)
    return m_lim, F_lim

m3, F3 = limiting_mag_snr(corrected_ZP, sigma_total_ap, exptime_s, 3)
m5, F5 = limiting_mag_snr(corrected_ZP, sigma_total_ap, exptime_s, 5)

print("\n=== Limiting Magnitudes with Read Noise Included ===")
print(f"Sky sigma = {sigma_sky_adu:.3f} ADU")
print(f"Read noise = {read_noise_adu:.2f} ADU")
print(f"Aperture N_pix = {N_pix}")
print(f"SNR=3: F_lim={F3:.3e} counts/s  -> m_lim={m3:.3f}")
print(f"SNR=5: F_lim={F5:.3e} counts/s  -> m_lim={m5:.3f}")
