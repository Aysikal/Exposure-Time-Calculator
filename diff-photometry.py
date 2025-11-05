import numpy as np
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval, ImageNormalize

# === User settings ===
fits_path = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\rezaei_saba_farideH_2025_10_22\GRB251013c\high\hot pixels removed\aligned\stacked\stacked-sumRGB.fits"
exptime_s = 6739.0  # seconds, integrated total exposure
print(f"Using fixed total exposure time (s) = {exptime_s}")

# approximate pixel coordinates (x, y)
targets = {
    "GRB_251013C": (2090.9, 1497.5),
    "Ref_star": (2372.0, 2110.0)
}

ref_known_mag = 19.20388806  # known magnitude of reference star (STAR 95 IN GRB PS1)
aperture_radius = 10    # pixels
inner_annulus = 15      # pixels
outer_annulus = 20      # pixels

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

def measure_flux(image, x, y, r, rin, rout):
    """Return net flux inside aperture after background subtraction."""
    cutout = Cutout2D(image, (x, y), (2*rout, 2*rout), mode='partial')
    cx, cy = cutout.to_cutout_position((x, y))
    img = cutout.data

    ap_mask = circular_mask(img.shape, (cx, cy), r)
    ann_mask = annulus_mask(img.shape, (cx, cy), rin, rout)

    ap_values = img[ap_mask]
    ann_values = img[ann_mask]

    # background from sigma-clipped annulus
    mean_bg, med_bg, std_bg = sigma_clipped_stats(ann_values, sigma=3.0)

    sum_ap = np.nansum(ap_values)
    n_pix = np.count_nonzero(ap_mask)
    net_flux = sum_ap - n_pix * med_bg
    return net_flux, med_bg, n_pix, img, (cx, cy)

# === Measure fluxes ===
results = {}
for name, (x, y) in targets.items():
    net, bg, n, img, center = measure_flux(data, x, y, aperture_radius, inner_annulus, outer_annulus)
    results[name] = {"net": net, "bg": bg, "n_pix": n, "center": center, "img": img}
    print(f"{name}: net={net:.2f}, background={bg:.2f}")

# === Differential photometry ===
if "GRB_251013C" in results and "Ref_star" in results:
    F_target = results["GRB_251013C"]["net"]
    F_ref = results["Ref_star"]["net"]

    if F_target > 0 and F_ref > 0:
        calibrated_mag = ref_known_mag - 2.5 * np.log10(F_target / F_ref)
        print(f"\nCalibrated magnitude for GRB_251013C = {calibrated_mag:.3f}")
    else:
        print("\nError: fluxes are not positive, cannot compute magnitude.")
else:
    print("\nMissing GRB or Ref_star in target list!")

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
