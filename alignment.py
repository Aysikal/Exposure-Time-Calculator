import os
import astroalign as aa
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval

# --- CONFIG ---
INPUT_FOLDER = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\reduced\October 1st, area 95 green high\keep"
OUTPUT_FOLDER = os.path.join(INPUT_FOLDER, "aligned")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

zscale = ZScaleInterval()

# --- I/O helpers ---
def load_fits(path):
    with fits.open(path, memmap=True) as hdul:
        data = hdul[0].data.astype(float)
        header = hdul[0].header
    return data, header

def save_fits(path, data, header=None):
    if header is None:
        hdu = fits.PrimaryHDU(data)
    else:
        hdu = fits.PrimaryHDU(data, header=header)
    hdu.writeto(path, overwrite=True)

def main():
    files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.fits', '.fit'))])
    if not files:
        raise SystemExit("No FITS files found.")

    # Reference frame
    ref_path = os.path.join(INPUT_FOLDER, files[0])
    ref_data, ref_hdr = load_fits(ref_path)
    save_fits(os.path.join(OUTPUT_FOLDER, files[0]), ref_data, ref_hdr)

    aligned_stack = [ref_data.copy().astype(float)]

    for i, fname in enumerate(files[1:], start=2):
        fpath = os.path.join(INPUT_FOLDER, fname)
        data, hdr = load_fits(fpath)
        try:
            aligned, footprint = aa.register(data, ref_data, detection_sigma=2.5)
            save_fits(os.path.join(OUTPUT_FOLDER, fname), aligned, hdr)
            aligned_stack.append(aligned.astype(float))
            print(f"[{i}/{len(files)}] {fname} aligned and saved.")
        except Exception as e:
            print(f"[{i}/{len(files)}] {fname} alignment failed: {e}")
            # Option: save original with history entry so output folder has one file per input
            hdr_fail = hdr.copy()
            hdr_fail.add_history(f"Alignment to {files[0]} FAILED; error: {e}")
            save_fits(os.path.join(OUTPUT_FOLDER, fname), data, hdr_fail)

    # --- Stack and plot ---
    if len(aligned_stack) == 0:
        print("No aligned frames to stack.")
        return

    stacked = np.median(np.stack(aligned_stack, axis=0), axis=0)
    stacked = np.nan_to_num(stacked)

    stacked_path = os.path.join(OUTPUT_FOLDER, "stacked.fits")
    save_fits(stacked_path, stacked.astype(ref_data.dtype), header=ref_hdr)
    print(f"Stacked image saved to: {stacked_path}")

    vmin, vmax = zscale.get_limits(stacked)
    plt.figure(figsize=(10, 8))
    plt.imshow(stacked, cmap='gray', origin='lower', vmin=vmin, vmax=vmax, interpolation='none')
    plt.title("Stacked Aligned Image (Median)")
    plt.colorbar(label="Pixel value")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
