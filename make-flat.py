import numpy as np
from astropy.io import fits
from astropy.convolution import convolve
from astropy.stats import sigma_clip
import matplotlib.pyplot as plt

# =========== Parameters ===========
science_path = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\Oct01\oct01_2025\target3\g\high\area95_g_T10C_2025_10_02_2x2_exp00.01.00.000_000001_High_1.fit"
n_boxes_target = 200
box_half = 1                  # 3x3 -> half-size = 1
min_sep_pix = 3               # exclude region within this many pixels from detected stars
threshold_sigma = 4         # threshold above background to call a star
dilation_size = 4 * min_sep_pix + 1
poly_degree = 2               # 0=constant, 1=plane, 2=quadratic
random_seed = 42
max_trials = 20000            # safety cap on attempts when sampling positions

# Robust box statistic settings
use_sigma_clip = True
box_sigma_clip = 2.5
box_clip_iters = 2

# =========== Load image ===========
with fits.open(science_path) as hd:
    img = hd[0].data.astype(float)
ny, nx = img.shape

# =========== Build star mask ===========
median = np.nanmedian(img)
mad = np.nanmedian(np.abs(img - median))
sigma_est = 1.4826 * mad

star_mask = (img > median + threshold_sigma * sigma_est)

kernel = np.ones((dilation_size, dilation_size), dtype=float)
dilated = convolve(star_mask.astype(float), kernel, normalize_kernel=False)
dilated_mask = dilated > 0

# =========== Find valid center pixels ===========
y_coords, x_coords = np.where(~dilated_mask)
# filter out positions too close to edges for a (2*box_half+1) box
good = ((y_coords >= box_half) &
        (y_coords < ny - box_half) &
        (x_coords >= box_half) &
        (x_coords < nx - box_half))
y_valid = y_coords[good]
x_valid = x_coords[good]

if len(y_valid) == 0:
    raise RuntimeError("No background positions found: relax threshold or min_sep_pix.")

# =========== Sample centers (avoid duplicate centers) ===========
rng = np.random.default_rng(random_seed)
chosen_centers = []
chosen_idx_set = set()
trials = 0
while len(chosen_centers) < n_boxes_target and trials < max_trials:
    trials += 1
    # pick a random candidate index
    idx = rng.integers(len(y_valid))
    if idx in chosen_idx_set:
        continue
    y = int(y_valid[idx]); x = int(x_valid[idx])
    # optional: ensure boxes do not overlap already chosen centers (comment out to allow overlap)
    min_center_sep = 0  # set >0 if you want non-overlapping boxes (in pixels)
    if min_center_sep > 0:
        too_close = any((abs(y - yy) <= min_center_sep and abs(x - xx) <= min_center_sep) for yy, xx in chosen_centers)
        if too_close:
            continue
    chosen_idx_set.add(idx)
    chosen_centers.append((y, x))

if len(chosen_centers) < n_boxes_target:
    print(f"Warning: only found {len(chosen_centers)} background boxes (requested {n_boxes_target}).")

# =========== Extract box statistics ===========
box_vals = []
centers_final = []
for (y, x) in chosen_centers:
    patch = img[y - box_half: y + box_half + 1, x - box_half: x + box_half + 1].copy()
    if use_sigma_clip:
        clipped = sigma_clip(patch.flatten(), sigma=box_sigma_clip, maxiters=box_clip_iters, masked=True)
        if np.all(clipped.mask):
            val = np.nanmedian(patch)  # fallback
        else:
            val = np.nanmedian(clipped.data[~clipped.mask])
    else:
        val = np.nanmedian(patch)
    box_vals.append(val)
    centers_final.append((y, x))

box_vals = np.array(box_vals)
ys = np.array([c[0] for c in centers_final])
xs = np.array([c[1] for c in centers_final])

# =========== Build design matrix for 2D polynomial ===========
# Create list of polynomial terms up to degree poly_degree
def poly_terms(x, y, deg):
    terms = []
    for i in range(deg + 1):
        for j in range(deg + 1 - i):
            terms.append((x**i) * (y**j))
    return np.vstack(terms).T

# scale coordinates to [-1,1] for numerical stability
x_scale = (xs - nx/2) / (nx/2)
y_scale = (ys - ny/2) / (ny/2)
A = poly_terms(x_scale, y_scale, poly_degree)   # shape (n_points, n_terms)

# robust fit: we can weight by measurement variance if available; here use simple outlier rejection
# initial least-squares
coeffs, *_ = np.linalg.lstsq(A, box_vals, rcond=None)

# optional simple iterative sigma-clipped fit
res = A.dot(coeffs) - box_vals
mask_good = np.abs(res - np.median(res)) < 3.5 * np.std(res)
if mask_good.sum() >= (A.shape[1] + 1):
    coeffs, *_ = np.linalg.lstsq(A[mask_good], box_vals[mask_good], rcond=None)

# =========== Evaluate model on full image grid ===========
yy, xx = np.mgrid[0:ny, 0:nx]
xxs = (xx.ravel() - nx/2) / (nx/2)
yys = (yy.ravel() - ny/2) / (ny/2)
A_full = poly_terms(xxs, yys, poly_degree)
model_flat = A_full.dot(coeffs)
model_image = model_flat.reshape(ny, nx)

# =========== Residuals and diagnostics ===========
residual_image = img - model_image

# =========== Plots ===========
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
ax = axes.ravel()
vmin = np.percentile(img, 5); vmax = np.percentile(img, 99)
ax[0].imshow(img, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
ax[0].scatter(xs, ys, s=6, edgecolor='red', facecolor='none', linewidth=0.6)
ax[0].set_title('Original image with selected background centers')

ax[1].imshow(model_image, origin='lower', cmap='viridis')
ax[1].set_title(f'Fitted background model degree={poly_degree}')

ax[2].imshow(residual_image, origin='lower', cmap='RdBu', vmin=-np.std(residual_image)*3, vmax=np.std(residual_image)*3)
ax[2].set_title('Residual (image - model)')

# histogram of box residuals
ax[3].hist(box_vals - (A.dot(coeffs)), bins=60, color='k', alpha=0.7)
ax[3].axvline(0, color='r', lw=1)
ax[3].set_title('Histogram of box residuals')

plt.tight_layout()
plt.show()

gain_table = model_image / np.nanmedian(model_image)
plt.imshow(gain_table, origin="lower")
plt.title("gain table from light - clear")
plt.colorbar()
plt.show()

# =========== Export gain map to FITS ===========

gain_hdu = fits.PrimaryHDU(gain_table)
gain_hdu.header['COMMENT'] = "Gain table generated from background model and residuals target3_u_T10C_2025_10_01_2x2_exp00.02.00.000_000001_High_3.fit"
gain_hdu.header['AUTHOR'] = "AYSAN HEMMATI"
gain_hdu.header['DATE'] = fits.getval(science_path, 'DATE', default='Unknown')

gain_hdul = fits.HDUList([gain_hdu])
gain_output_path = science_path.replace('.fit', '_gain_table.fits')
gain_hdul.writeto(gain_output_path, overwrite=True)

