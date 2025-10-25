import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
FITS_FOLDER = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\data 2\Rezaei_Altafi_10_07_2025\standard_star\94 B2\clear\high\keep\reduced"
COORDS_NPY = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\Star Coords\standard_star\94 B2\oct7-clear-94B2.npy"

BOX_HALF_SIZE = 12  # pixels around star (total box 25x25)

# ---------------- HELPERS ----------------
def list_fits(folder):
    return sorted([os.path.join(folder, f) for f in os.listdir(folder)
                   if f.lower().endswith((".fit", ".fits"))])

def load_fits_data(path):
    with fits.open(path, memmap=True) as hdul:
        data = hdul[0].data.astype(float)
    return np.nan_to_num(data)

def extract_box(data, x, y, half_size):
    x, y = int(round(x)), int(round(y))
    y1, y2 = max(y - half_size, 0), min(y + half_size + 1, data.shape[0])
    x1, x2 = max(x - half_size, 0), min(x + half_size + 1, data.shape[1])
    return data[y1:y2, x1:x2]

# ---------------- MAIN ----------------
def main():
    # Load star coords
    coords = np.load(COORDS_NPY)
    # Assuming first row, columns 3 and 4 store x,y
    x_star, y_star = float(coords[0, 3]), float(coords[0, 2])
    print(f"Using star coords: x={x_star}, y={y_star}")

    # List FITS files
    fits_files = list_fits(FITS_FOLDER)
    if not fits_files:
        raise SystemExit("No FITS files found!")

    # Loop over FITS and show star
    for f in fits_files:
        data = load_fits_data(f)
        box = extract_box(data, x_star, y_star, BOX_HALF_SIZE)

        plt.figure(figsize=(4, 4))
        vmin = np.percentile(box, 5)
        vmax = np.percentile(box, 99)
        plt.imshow(box, origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
        plt.title(os.path.basename(f))
        plt.colorbar(label='Counts')
        plt.show()

if __name__ == "__main__":
    main()
