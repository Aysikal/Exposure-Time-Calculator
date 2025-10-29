import os
import numpy as np
from astropy.io import fits
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------
FILTER_FOLDERS = {
    "clear": r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\dark_corrected\low\clear",
    "g":     r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\dark_corrected\low\g",
    "r":     r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\dark_corrected\low\r",
    "i":     r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\dark_corrected\low\i"
}
STAR_COORD_FILES = {
    "clear": r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\Star Coords\extiction\oct 1\clear\area 92\oct1-clear-area92-star1.npy",
    "g":     r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\Star Coords\extiction\oct 1\g\area 92\oct1-g-area92-star1.npy",
    "r":     r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\Star Coords\extiction\oct 1\r\area 92\oct1-r-area92-star1.npy",
    "i":     r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\Star Coords\extiction\oct 1\i\area 92\oct1-i-area92-star1.npy"
}

STAR_NAME = "Area 92 star 7 (not the green tho)"
OUTPUT_FOLDER = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\Star Coords\extiction\area 92 eta"

# ------------------------------------------------------
# BOX SIZE
# ------------------------------------------------------
PSF_ARCSEC = 0.7
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
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data.astype(float)
        hdr = hdul[0].header
    return np.nan_to_num(data), hdr

def get_exptime_sec(header):
    if EXPTIME_KEY in header:
        return float(header[EXPTIME_KEY]) / 1e5
    

def extract_box(data, x, y, half_size):
    x, y = int(round(x)), int(round(y))
    y1, y2 = y - half_size, y + half_size + 1
    x1, x2 = x - half_size, x + half_size + 1
    y1, y2 = max(y1, 0), min(y2, data.shape[0])
    x1, x2 = max(x1, 0), min(x2, data.shape[1])
    print(f"  → Extracting box: x=[{x1}:{x2}], y=[{y1}:{y2}], shape={data[y1:y2, x1:x2].shape}")
    return data[y1:y2, x1:x2]

def load_star_coords(npy_path):
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"Missing star coord file: {npy_path}")
    coords = np.load(npy_path)
    print(f"\n[DEBUG] Loaded coord file: {npy_path}")
    print(f"  Shape: {coords.shape}")
    print(f"  First few rows:\n{coords[:5]}")
    return coords

def plot_median_box(image, filter_name, save_dir, x_center, y_center):
    plt.figure(figsize=(5, 4))
    vmin = np.percentile(image, 5)
    vmax = np.percentile(image, 99)

    half_y, half_x = image.shape[0] // 2, image.shape[1] // 2
    extent = [
        x_center - half_x, x_center + half_x,
        y_center - half_y, y_center + half_y
    ]

    plt.imshow(image, origin='lower', cmap='inferno',
               vmin=vmin, vmax=vmax, extent=extent)
    plt.colorbar(label='Counts/s')
    plt.title(f"Median box — {filter_name.upper()} filter")
    plt.xlabel('X pixels (original)')
    plt.ylabel('Y pixels (original)')
    plt.plot(x_center, y_center, 'ro', markersize=5, label='Star center')
    plt.legend(loc='upper right')

    plt.tight_layout()
    if SAVE_FIGS:
        out_path = os.path.join(save_dir, f"median_box_{filter_name}.png")
        plt.savefig(out_path, dpi=150)
        print(f"  Saved median box figure: {out_path}")
    #plt.show()

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
        coords = load_star_coords(STAR_COORD_FILES[filt])

        if len(coords) < len(fits_files):
            print(f"[WARNING] Coord file has fewer entries ({len(coords)}) than FITS files ({len(fits_files)}).")
        
        boxes = []
        for i, f in enumerate(fits_files):
            data, hdr = load_fits_data(f)
            exptime = get_exptime_sec(hdr)

            # --- Handle coordinates safely ---
            if i < len(coords):
                x_c3, y_c2 = coords[i, 3], coords[i, 2]
                x_c1, y_c0 = coords[i, 1], coords[i, 0]
                if np.isnan(x_c3) or np.isnan(y_c2):
                    x_star, y_star = x_c1, y_c0
                    print(f"  [INFO] NaN detected → using fallback coords (cols 1 & 2): ({x_star:.2f}, {y_star:.2f})")
                else:
                    x_star, y_star = x_c3, y_c2
            else:
                x_star = coords[-1, 1]
                y_star = coords[-1, 0]

            box = extract_box(data, x_star, y_star, BOX_HALF_SIZE)

            # --- FIX: Skip invalid (edge-cut) boxes ---
            if box.size == 0 or box.shape[0] != BOX_SIZE_PX or box.shape[1] != BOX_SIZE_PX:
                print(f"  [WARNING] Skipping frame {i} → invalid box shape {box.shape}")
                continue

            boxes.append(box / exptime)

        if not boxes:
            print(f"[ERROR] No valid boxes extracted for filter {filt}")
            continue

        median_box = np.median(np.stack(boxes), axis=0)
        flux = np.sum(median_box)
        flux_results[filt] = flux
        print(f"Computed flux = {flux:.3e} counts/s")

        mid_index = len(coords) // 2
        x_med = float(coords[mid_index, 1])
        y_med = float(coords[mid_index, 0])
        plot_median_box(median_box, filt, folder, x_med, y_med)

    # --- η computation ---
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

    df = pd.DataFrame(results)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    out_csv = os.path.join(OUTPUT_FOLDER, f"eta_results_{STAR_NAME}.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved results → {out_csv}")
    print(df)

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
