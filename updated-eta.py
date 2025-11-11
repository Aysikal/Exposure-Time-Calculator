import os
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import pandas as pd
from astropy.nddata.utils import Cutout2D
from scipy.optimize import curve_fit

# ---------------- CONFIGURATION ---------------- #
FILTER_FOLDERS = {
    "clear": r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\Rezaei_Hossein_Atanaz_Kosar_2025_11_04\light\Aysan\high\hot pixels removed\97b-8\aligned\clear\dark_corrected",
    "g":     r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\Rezaei_Hossein_Atanaz_Kosar_2025_11_04\light\Aysan\high\hot pixels removed\97b-8\aligned\g\dark_corrected",
    "r":     r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\Rezaei_Hossein_Atanaz_Kosar_2025_11_04\light\Aysan\high\hot pixels removed\97b-8\aligned\r\dark_corrected",
    "i":     r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\Rezaei_Hossein_Atanaz_Kosar_2025_11_04\light\Aysan\high\hot pixels removed\97b-8\aligned\i\dark_corrected"
}

OUTPUT_FOLDER = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\Star Coords\aligned_eta"
PSF_ARCSEC = 1
BOX_FACTOR = 10.0
PIXEL_SCALE = 0.101  # arcsec/pixel
BOX_SIZE_PX = round((BOX_FACTOR * PSF_ARCSEC) / PIXEL_SCALE)
if BOX_SIZE_PX % 2 == 0:
    BOX_SIZE_PX += 1
BOX_HALF_SIZE = BOX_SIZE_PX // 2

EXPTIME_KEY = "EXPTIME"
SAVE_FIGS = True

# ---------------- HELPER FUNCTIONS ---------------- #
def list_fits(folder):
    return sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".fit")])

def load_fits_data(path):
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data.astype(float)
        hdr = hdul[0].header
    return np.nan_to_num(data), hdr

def stack_fits(folder):
    fits_files = list_fits(folder)
    if not fits_files:
        raise FileNotFoundError(f"No FITS files found in {folder}")
    data_list = []
    for f in fits_files:
        data, hdr = load_fits_data(f)
        exptime = float(hdr.get(EXPTIME_KEY, 1.0)) * 1e-5  # scale exptime if needed
        data_list.append(data / exptime)
    stack = np.median(np.stack(data_list), axis=0)
    return stack, fits_files[0]

def onclick(event):
    if event.inaxes:
        x, y = event.xdata, event.ydata
        star_coords.append((x, y))
        event.inaxes.plot(x, y, 'rx')
        event.canvas.draw()

def extract_box(data, x, y, half_size):
    cutout = Cutout2D(data, (x, y), size=(half_size*2+1, half_size*2+1))
    return cutout.data

# 2D Gaussian model
def gaussian_2d(coords, amp, x0, y0, sigma_x, sigma_y, offset):
    x, y = coords
    g = offset + amp * np.exp(-(((x-x0)**2)/(2*sigma_x**2) + ((y-y0)**2)/(2*sigma_y**2)))
    return g.ravel()

def refine_center(box):
    y_size, x_size = box.shape
    x = np.arange(x_size)
    y = np.arange(y_size)
    X, Y = np.meshgrid(x, y)
    initial_guess = (box.max(), x_size//2, y_size//2, 1.0, 1.0, np.median(box))
    try:
        popt, _ = curve_fit(gaussian_2d, (X, Y), box.ravel(), p0=initial_guess)
        _, x0, y0, _, _, _ = popt
        return x0, y0
    except:
        return x_size//2, y_size//2  # fallback if fit fails

# ---------------- MAIN ---------------- #
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
star_coords = []


# Stack all filters and save
STACKED_FOLDER = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\Rezaei_Hossein_Atanaz_Kosar_2025_11_04\light\Aysan\high\hot pixels removed\97b-8\aligned\stacked"
os.makedirs(STACKED_FOLDER, exist_ok=True)
stacks = {}

for filt, folder in FILTER_FOLDERS.items():
    print(f"Stacking {filt} filter...")
    stack_data, ref_file = stack_fits(folder)
    stacks[filt] = stack_data
    
    # Save stacked FITS
    ref_hdr = fits.getheader(ref_file)
    stack_path = os.path.join(STACKED_FOLDER, f"{filt}_stack.fits")
    fits.writeto(stack_path, stack_data, ref_hdr, overwrite=True)
    print(f"Saved stacked {filt} → {stack_path}")

# Select stars on clear stack
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.imshow(stacks["clear"], cmap='gray', origin='lower',
          vmin=np.percentile(stacks["clear"],5),
          vmax=np.percentile(stacks["clear"],95))
ax.set_title("Click on stars, then close the window")
fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

print(f"Selected stars (approx): {star_coords}")

# Extract fluxes and compute eta with refined centers
results = []
for i, (x, y) in enumerate(star_coords):
    flux_clear = None
    star_result = {"Star": i+1}
    # Refine in clear image
    box_clear = extract_box(stacks["clear"], x, y, BOX_HALF_SIZE)
    dx, dy = refine_center(box_clear)
    x_refined = x - BOX_HALF_SIZE + dx
    y_refined = y - BOX_HALF_SIZE + dy
    # Clear flux
    flux_clear = np.sum(extract_box(stacks["clear"], x_refined, y_refined, BOX_HALF_SIZE))
    star_result["Flux_clear"] = flux_clear
    # Fluxes in other filters
    for filt in ["g","r","i"]:
        box_filt = extract_box(stacks[filt], x_refined, y_refined, BOX_HALF_SIZE)
        flux_filt = np.sum(box_filt)
        eta = flux_filt / flux_clear
        star_result[f"Flux_{filt}"] = flux_filt
        star_result[f"Eta_{filt}"] = eta
    results.append(star_result)

# Save results
df = pd.DataFrame(results)
csv_path = os.path.join(OUTPUT_FOLDER, "eta_results_aligned_stars_refined.csv")
df.to_csv(csv_path, index=False)
print(f"Saved results → {csv_path}")
print(df)

# Compute median eta and error for each filter
medians = {}
for filt in ["g","r","i"]:
    etas = df[f"Eta_{filt}"].values
    medians[filt] = (np.median(etas), np.std(etas))
print("\nMedian η ± std for each filter:")
for filt in ["g","r","i"]:
    print(f"{filt}: {medians[filt][0]:.3f} ± {medians[filt][1]:.3f}")

# Plot eta barplots for each star
for i, row in df.iterrows():
    plt.figure(figsize=(6,4))
    etas = [row[f"Eta_{filt}"] for filt in ["g","r","i"]]
    plt.bar(["g","r","i"], etas, color=['limegreen','gold','tomato'])
    plt.ylabel("η = Flux(filter) / Flux(clear)")
    plt.title(f"Star {int(row['Star'])}")
    for j, val in enumerate(etas):
        plt.text(j, val+0.01, f"{val:.3f}", ha='center', va='bottom', fontsize=9)
    if SAVE_FIGS:
        plt.savefig(os.path.join(OUTPUT_FOLDER, f"eta_star{int(row['Star'])}.png"), dpi=150)
    plt.show()
