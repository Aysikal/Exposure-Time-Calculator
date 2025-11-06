import os
import sys
import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval, ImageNormalize
import matplotlib.pyplot as plt

def list_fits(folder):
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith((".fit", ".fits"))
    )

def load_fits(path):
    with fits.open(path, memmap=False, do_not_scale_image_data=False) as hdul:
        for hdu in hdul:
            if hdu.data is not None:
                data = hdu.data
                break
        else:
            raise ValueError(f"No image data found in {path}")

        # Convert masked arrays or non-finite values to np.nan
        if isinstance(data, np.ma.MaskedArray):
            data = data.filled(np.nan)
        data = np.asarray(data, dtype=float)
        data[~np.isfinite(data)] = np.nan

    return data

def stack_sum(fits_list):
    stack = []
    bad_files = []

    for f in fits_list:
        try:
            data = load_fits(f)
            stack.append(data)
        except Exception as e:
            print(f"Skipping {os.path.basename(f)}: {e}", file=sys.stderr)
            bad_files.append((f, str(e)))

    if not stack:
        raise ValueError("No valid FITS images to stack.")

    arr = np.stack(stack, axis=0)
    result = np.nansum(arr, axis=0)

    return result, bad_files

def plot_image(image, title="Stacked", cmap="gray_r", figsize=(8,6), savepath=None):
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(image[np.isfinite(image)])
    norm = ImageNormalize(vmin=vmin, vmax=vmax)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(image, origin="lower", cmap=cmap, norm=norm)
    ax.set_title(title)
    ax.set_xlabel("X (pix)")
    ax.set_ylabel("Y (pix)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Counts per second")
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200)
    plt.show()

if __name__ == "__main__":
    FOLDER = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\rezaei_saba_farideH_2025_10_22\GRB251013c\high\hot pixels removed\aligned\reduced"
    fits_list = list_fits(FOLDER)
    print(f"Found {len(fits_list)} FITS files.")

    sum_stack, bad = stack_sum(fits_list)
    print(f"Sum stack shape: {sum_stack.shape}; skipped {len(bad)} files.")

    # Save plot
    out_png = os.path.join(FOLDER, "stacked_sum.png")
    plot_image(sum_stack, title=f"Sum stack of {len(fits_list)-len(bad)} files", savepath=out_png)

    # Save FITS with integrated exposure in header
    out_fits = os.path.join(FOLDER, "stacked_sum.fits")
    hdr = fits.Header()
    hdr["TOTAL_EXPT"] = 6739  # total integrated exposure in seconds
    hdr["STACK"] = "SUM"
    fits.writeto(out_fits, sum_stack, header=hdr, overwrite=True)
    print("Wrote:", out_fits)
