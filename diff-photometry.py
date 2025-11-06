import numpy as np
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval, ImageNormalize

# === User settings ===
fits_path = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\rezaei_saba_farideH_2025_10_22\GRB251013c\high\hot pixels removed\aligned\reduced\stacked_sum.fits"
exptime_s = 6739.0  # seconds, integrated total exposure
print(f"Using fixed total exposure time (s) = {exptime_s}")

# approximate pixel coordinates (x, y)
targets = {
    "GRB_251013C": (2090.9, 1497.5),
    "Ref_star_95": (2370.0, 2110.0), 
    "Ref_star_82" : (2807, 2759),
    "Ref_star_26" : (2629, 1472),
    "Ref_star_97" : (2384, 774),
    "Ref_star_4" : (2079, 662),
    "Ref_star_21" : (713, 2548)
    # Add more reference stars here if needed
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

# === Load FITS data ===
data = fits.getdata(fits_path)
hdr = fits.getheader(fits_path)

# === Functions ===
def circular_mask(shape, center, radius):
    yy, xx = np.indices(shape)
    cx, cy = center
    return (xx - cx)**2 + (yy - cy)**2 <= radius**2

def annulus_mask(shape, center, r_in, r_out):
    yy, xx = np.indices(shape)
    cx, cy = center
    r2 = (xx - cx)**2 + (yy - cy)**2
    return (r2 >= r_in**2) & (r2 <= r_out**2)

def measure_flux(image, x, y, r, rin, rout, exptime):
    """Return normalized (per-second) net flux inside aperture after background subtraction."""
    cutout = Cutout2D(image, (x, y), (2*rout, 2*rout), mode='partial')
    cx, cy = cutout.to_cutout_position((x, y))
    img = cutout.data

    ap_mask = circular_mask(img.shape, (cx, cy), r)
    ann_mask = annulus_mask(img.shape, (cx, cy), rin, rout)

    ap_values = img[ap_mask]
    ann_values = img[ann_mask]

    mean_bg, med_bg, std_bg = sigma_clipped_stats(ann_values, sigma=3.0)

    sum_ap = np.nansum(ap_values)
    n_pix = np.count_nonzero(ap_mask)
    net_flux = sum_ap - n_pix * med_bg

    # Normalize by exposure time
    normalized_flux = net_flux / exptime

    return normalized_flux, med_bg, n_pix, img, (cx, cy)

# === Measure fluxes ===
results = {}
print("\n=== Normalized Instrumental Flux Measurements (per second) ===")
for name, (x, y) in targets.items():
    net, bg, n, img, center = measure_flux(data, x, y, aperture_radius, inner_annulus, outer_annulus, exptime_s)
    results[name] = {"net": net, "bg": bg, "n_pix": n, "center": center, "img": img}
    inst_mag = -2.5 * np.log10(net) if net > 0 else np.nan
    print(f"{name}: net_flux = {net:.4e} counts/s, background = {bg:.2f}, n_pix = {n}, instrumental mag = {inst_mag:.3f}")

# === Differential photometry with multiple reference stars ===
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
                print(f"Using {ref_name}: GRB_mag = {mag:.3f} (Ref mag = {ref_mag:.3f}, F_ref = {F_ref:.4e} counts/s)")
            else:
                print(f"Skipping {ref_name}: non-positive flux for GRB or reference star.")

    if mag_list:
        mean_mag = np.mean(mag_list)
        std_mag = np.std(mag_list)
        print(f"\nMean calibrated magnitude for GRB_251013C = {mean_mag:.3f} ± {std_mag:.3f}")
else:
    print("\nMissing GRB or reference stars in target list!")

# === Optional visualization ===
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
