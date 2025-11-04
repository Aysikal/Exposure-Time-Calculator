import numpy as np
import matplotlib.pyplot as plt
import pywt

# Recreate the same 100x100 image with 30 random hot pixels (seeded for reproducibility)
np.random.seed(42)
size = 100
num_hot = 30
img = np.zeros((size, size), dtype=float)
idx = np.random.choice(img.size, size=num_hot, replace=False)
r, c = np.unravel_index(idx, img.shape)
img[r, c] = 1.0

# Helper: one-level dwt2 + idwt2 with soft thresholding on detail coeffs
def denoise_onelevel(img, wavelet_name, threshold):
    cA, (cH, cV, cD) = pywt.dwt2(img, wavelet_name, mode='periodization')
    # Soft-threshold detail coefficients
    def soft(t, x):
        return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)
    cH_t = soft(threshold, cH)
    cV_t = soft(threshold, cV)
    cD_t = soft(threshold, cD)
    rec = pywt.idwt2((cA, (cH_t, cV_t, cD_t)), wavelet_name, mode='periodization')
    return (cA, cH, cV, cD), (cA, cH_t, cV_t, cD_t), rec

# Choose thresholds (try small values; adjust interactively)
threshold = 0.5

wavelets = ['db1', 'db2']
results = {}
for w in wavelets:
    raw_coeffs, thresh_coeffs, rec = denoise_onelevel(img, w, threshold)
    results[w] = {'raw': raw_coeffs, 'thresh': thresh_coeffs, 'rec': rec}

# Plot original and results
fig, axes = plt.subplots(3, len(wavelets) + 1, figsize=(12, 9))
plt.subplots_adjust(wspace=0.35, hspace=0.4)

# Column 0: original
axes[0, 0].imshow(img, cmap='hot', origin='lower')
axes[0, 0].set_title('Original (100x100)')
axes[0, 0].axis('off')

axes[1, 0].axis('off')
axes[2, 0].axis('off')

for col, w in enumerate(wavelets, start=1):
    # Row 0: reconstructed image after thresholding
    axes[0, col].imshow(results[w]['rec'], cmap='hot', origin='lower', vmin=0, vmax=1)
    axes[0, col].set_title(f'{w} denoised')
    axes[0, col].axis('off')

    # Row 1: raw coeff grids (show cA, cH, cV, cD concatenated as inset image)
    cA, cH, cV, cD = results[w]['raw']
    # Normalize each for display and tile into small mosaic
    def norm(x):
        mm = np.max(np.abs(x))
        return x / (mm or 1.0)
    small_tiles = np.block([[norm(cA), norm(cH)], [norm(cV), norm(cD)]])
    axes[1, col].imshow(small_tiles, cmap='seismic', origin='lower')
    axes[1, col].set_title(f'{w} raw coeffs (A H; V D)')
    axes[1, col].axis('off')

    # Row 2: thresholded coeff grids
    cA_t, cH_t, cV_t, cD_t = results[w]['thresh']
    small_tiles_t = np.block([[norm(cA_t), norm(cH_t)], [norm(cV_t), norm(cD_t)]])
    axes[2, col].imshow(small_tiles_t, cmap='seismic', origin='lower')
    axes[2, col].set_title(f'{w} thresh coeffs')
    axes[2, col].axis('off')

# Add a colorbar to the leftmost reconstructed plots
cax = fig.add_axes([0.92, 0.55, 0.015, 0.3])
plt.colorbar(plt.cm.ScalarMappable(cmap='hot'), cax=cax, label='Intensity')

fig.suptitle('Single-level wavelet denoising: db1 vs db2 (soft threshold on details)')
plt.show()
