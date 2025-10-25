import os
import numpy as np
from astropy.io import fits
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------
FILTER_FOLDERS = {
    "clear": r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\data 2\Rezaei_Altafi_10_07_2025\standard_star\94 B2\clear\high\keep\reduced",
    "g":     r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\data 2\Rezaei_Altafi_10_07_2025\standard_star\94 B2\g\high\keep\reduced",
    "r":     r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\data 2\Rezaei_Altafi_10_07_2025\standard_star\94 B2\r\high\keep\reduced",
    "i":     r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\data 2\Rezaei_Altafi_10_07_2025\standard_star\94 B2\i\high\keep\reduced"
}
STAR_COORD_FILES = {
    "clear": r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\Star Coords\standard_star\94 B2\oct7-clear-94B2.npy",
    "g":     r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\Star Coords\standard_star\94 B2\oct7-g-94B2.npy",
    "r":     r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\Star Coords\standard_star\94 B2\oct7-r-94B2.npy",
    "i":     r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\Star Coords\standard_star\94 B2\oct7-i-94B2.npy"
}

STAR_NAME = "94B2"
OUTPUT_FOLDER = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\Star Coords\standard_star\94 B2"


# Updated: dynamically match the box size used in .npy creation
PSF_ARCSEC = 1
PIXEL_SCALE = 0.047 * 1.8
BOX_FACTOR = 10.0
BOX_SIZE_PX = round((BOX_FACTOR * PSF_ARCSEC) / PIXEL_SCALE)
if BOX_SIZE_PX % 2 == 0:
    BOX_SIZE_PX += 1
BOX_HALF_SIZE = BOX_SIZE_PX // 2

EXPTIME_KEY = "EXPTIME"
SAVE_FIGS = True

# ------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------
def list_fits(folder):
    return sorted([os.path.join(folder, f) for f in os.listdir(folder)
                   if f.lower().endswith((".fit", ".fits"))])

def load_fits_data(path):
    with fits.open(path, memmap=True) as hdul:
        data = hdul[0].data.astype(float)
        hdr = hdul[0].header
    return np.nan_to_num(data), hdr

def get_exptime_sec(header):
    if EXPTIME_KEY in header:
        return float(header[EXPTIME_KEY]) / 1e5
    return 1.0  # fallback if key missing

def extract_box(data, x, y, half_size):
    x, y = int(round(x)), int(round(y))
    y1, y2 = y - half_size, y + half_size + 1
    x1, x2 = x - half_size, x + half_size + 1
    y1, y2 = max(y1, 0), min(y2, data.shape[0])
    x1, x2 = max(x1, 0), min(x2, data.shape[1])
    return data[y1:y2, x1:x2]

def load_star_coords(npy_path):
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"Missing star coord file: {npy_path}")
    coords = np.load(npy_path)
    x_star, y_star = coords[0, 3], coords[0, 2]
    return float(x_star), float(y_star)

def plot_median_box(image, filter_name, save_dir):
    plt.figure(figsize=(5, 4))
    # Use robust scaling to make faint stars visible
    vmin = np.percentile(image, 5)
    vmax = np.percentile(image, 99)
    plt.imshow(image, origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Counts/s')
    plt.title(f"Median box — {filter_name.upper()} filter")
    plt.xlabel('X pixels')
    plt.ylabel('Y pixels')
    plt.tight_layout()
    if SAVE_FIGS:
        out_path = os.path.join(save_dir, f"median_box_{filter_name}.png")
        plt.savefig(out_path, dpi=150)
    plt.show()

# ------------------------------------------------------
# MAIN
# ------------------------------------------------------
def main():
    flux_results = {}

    for filt, folder in FILTER_FOLDERS.items():
        fits_files = list_fits(folder)
        if not fits_files:
            print(f"[WARNING] No FITS files found for {filt}")
            continue

        print(f"\n=== {filt.upper()} FILTER ({len(fits_files)} frames) ===")
        x_star, y_star = x_star, y_star = load_star_coords(STAR_COORD_FILES[filt])

        print(f"Star coords from npy: x={x_star:.2f}, y={y_star:.2f}")

        # Quick check: plot the star on the first frame
        first_data, _ = load_fits_data(fits_files[0])
        plt.figure(figsize=(5, 5))
        plt.imshow(first_data, origin='lower', cmap='inferno',
                   vmin=np.percentile(first_data, 5),
                   vmax=np.percentile(first_data, 99))
        plt.plot(x_star, y_star, 'ro', markersize=6)
        plt.title(f"Star check — {filt.upper()} filter")
        plt.show()

        boxes = []
        for f in fits_files:
            data, hdr = load_fits_data(f)
            exptime = get_exptime_sec(hdr)
            box = extract_box(data, x_star, y_star, BOX_HALF_SIZE)
            boxes.append(box / exptime)

        median_box = np.median(np.stack(boxes), axis=0)
        flux = np.sum(median_box)
        flux_results[filt] = flux

        # Save median box FITS
        print(f"flux = {flux:.3e} counts/s")

        # Plot median box
        plot_median_box(median_box, filt, folder)

    # Compute η relative to CLEAR
    if "clear" not in flux_results:
        raise ValueError("No CLEAR data found — cannot compute η.")

    clear_flux = flux_results["clear"]
    results = []
    for filt, flux in flux_results.items():
        if filt == "clear":
            continue
        eta = flux / clear_flux
        results.append({
            "Filter": filt,
            "Flux_filter": flux,
            "Flux_clear": clear_flux,
            "Eta": eta
        })
        print(f"η({filt}) = {eta:.4f}")

    # Save to CSV
    df = pd.DataFrame(results)
    out_csv = os.path.join(OUTPUT_FOLDER, f"eta_results_{STAR_NAME}.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved results → {out_csv}")
    print(df)

    # Plot η bar chart
    plt.figure(figsize=(6, 4))
    plt.bar(df["Filter"], df["Eta"], color=['limegreen', 'gold', 'tomato'])
    plt.ylabel("η = Flux(filter) / Flux(clear)")
    plt.title(f"Transmission Efficiency (η) — Star {STAR_NAME}")
    for i, val in enumerate(df["Eta"]):
        plt.text(i, val + 0.01, f"{val:.3f}", ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig(os.path.join(OUTPUT_FOLDER, f"eta_barplot_{STAR_NAME}.png"), dpi=150)
    plt.show()

# ------------------------------------------------------
if __name__ == "__main__":
    main()
