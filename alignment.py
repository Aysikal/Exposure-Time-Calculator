import os
import astroalign as aa
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval

# --- CONFIG ---
INPUT_FOLDER = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\reduced\sept 30 area 95 g low\keep"
OUTPUT_FOLDER = os.path.join(INPUT_FOLDER, "aligned")
DIAGNOSTIC_FOLDER = os.path.join(OUTPUT_FOLDER, "diagnostics")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(DIAGNOSTIC_FOLDER, exist_ok=True)

zscale = ZScaleInterval()

# --- I/O helpers ---
def load_fits(path):
    with fits.open(path, memmap=True) as hdul:
        data = hdul[0].data.astype(float)
        header = hdul[0].header
    return data, header

def save_fits(path, data, header=None):
    hdu = fits.PrimaryHDU(data, header=header) if header else fits.PrimaryHDU(data)
    hdu.writeto(path, overwrite=True)

def save_overlay(ref, aligned, fname, alpha=0.5):
    vmin, vmax = zscale.get_limits(ref)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(ref, cmap='gray', origin='lower', vmin=vmin, vmax=vmax, alpha=alpha)
    ax.imshow(aligned, cmap='hot', origin='lower', vmin=vmin, vmax=vmax, alpha=alpha)
    ax.set_title(f"Overlay: {fname}")
    plt.tight_layout()
    overlay_path = os.path.join(DIAGNOSTIC_FOLDER, f"{fname}_overlay.png")
    plt.savefig(overlay_path)
    plt.close()

def main():
    files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.fits', '.fit'))])
    if not files:
        raise SystemExit("No FITS files found.")

    ref_path = os.path.join(INPUT_FOLDER, files[0])
    ref_data, ref_hdr = load_fits(ref_path)
    save_fits(os.path.join(OUTPUT_FOLDER, files[0]), ref_data, ref_hdr)

    aligned_stack = [ref_data.copy().astype(float)]

    for i, fname in enumerate(files[1:], start=2):
        fpath = os.path.join(INPUT_FOLDER, fname)
        data, hdr = load_fits(fpath)
        try:
            aligned, footprint = aa.register(data, ref_data, detection_sigma=1.2)
            coverage = np.sum(footprint) / footprint.size
            print(f"[{i}/{len(files)}] {fname} aligned. Footprint coverage: {coverage:.1%}")

            if coverage < 0.5:
                raise ValueError(f"Low overlap ({coverage:.1%})")

            save_fits(os.path.join(OUTPUT_FOLDER, fname), aligned, hdr)
            aligned_stack.append(aligned.astype(float))
            save_overlay(ref_data, aligned, fname)

        except Exception as e:
            print(f"[{i}/{len(files)}] {fname} alignment failed: {e}")
            hdr_fail = hdr.copy()
            hdr_fail.add_history(f"Alignment to {files[0]} FAILED; error: {e}")
            save_fits(os.path.join(OUTPUT_FOLDER, fname), data, hdr_fail)

    if len(aligned_stack) <= 1:
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
