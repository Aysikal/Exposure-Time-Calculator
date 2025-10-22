import os
import sys
import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval, ImageNormalize
import matplotlib.pyplot as plt

EXPTIME_KEYS = ["EXPTIME", "EXPOSURE", "EXPTM", "EXPTIM", "TELAPSE"]

def list_fits(folder):
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith((".fit", ".fits"))
    )

from astropy.io import fits
import numpy as np

def load_fits(path):
    """
    Robust loader for PixInsight-produced FITS:
    - uses memmap=False so files with BZERO/BSCALE/BLANK open correctly
    - lets astropy apply scaling (do_not_scale_image_data=False)
    - converts masked arrays and non-finite values to np.nan
    - returns (data: 2D float ndarray, header)
    """
    with fits.open(path, memmap=False, do_not_scale_image_data=False) as hdul:
        # find first HDU that contains image data
        for hdu in hdul:
            if hdu.data is None:
                continue
            data = hdu.data
            hdr = hdu.header
            break
        else:
            raise ValueError(f"No image data found in any HDU of {path}")

        # If astropy returned a MaskedArray it will already respect BLANK; fill with nan
        if isinstance(data, np.ma.MaskedArray):
            data = data.filled(np.nan)

        # Ensure float dtype and replace inf with nan
        data = np.asarray(data, dtype=float)
        data[~np.isfinite(data)] = np.nan

    return data, hdr

def get_exptime(header):
    for key in EXPTIME_KEYS:
        if key in header:
            try:
                return float(header[key])
            except Exception:
                continue
    raise KeyError(f"Exposure time not found in header keys: {EXPTIME_KEYS}")


def stack_median(fits_list, load_fits, get_exptime=None, skip_bad=True):
    """
    Median-stack FITS images without normalizing by exposure time.
    - fits_list: iterable of file paths
    - load_fits: callable(path) -> (data, header)
    - get_exptime: optional callable(header) -> float (used only for validation if provided)
    - skip_bad: skip files that fail to load or fail validation; collect them in bad_files
    Returns: (median_image, bad_files) where bad_files is a list of (path, error_str)
    """
    stack = []
    bad_files = []
    for f in fits_list:
        try:
            data, hdr = load_fits(f)
            # optional validation of exposure time (keeps original behaviour if desired)
            if get_exptime is not None:
                exptime = get_exptime(hdr)
                if exptime <= 0 or not np.isfinite(exptime):
                    raise ValueError(f"Invalid EXPTIME ({exptime})")
            # keep NaNs so np.nanmedian can handle pixels missing in some frames
            stack.append(np.asarray(data, dtype=float))
        except Exception as e:
            msg = f"Skipping {os.path.basename(f)}: {e}"
            if skip_bad:
                print(msg, file=sys.stderr)
                bad_files.append((f, str(e)))
                continue
            else:
                raise
    if not stack:
        raise ValueError("No valid FITS images to stack after filtering.")
    arr = np.stack(stack, axis=0)           # shape (N, ny, nx)
    med = np.nanmedian(arr, axis=0)        # median along the stack axis
    return med, bad_files


def plot_image(image, title="Stacked (counts/s)", cmap="gray_r", figsize=(8,6), savepath=None):
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
    FOLDER = r"C:\Users\AYSAN\Desktop\Crab\g\high\reduced"
    fits_list = list_fits(FOLDER)
    print(f"Found {len(fits_list)} FITS files.")
    # pass load_fits and optionally get_exptime into stack_median
    stacked, bad = stack_median(fits_list, load_fits, get_exptime=None, skip_bad=True)
    print(f"Stack shape: {stacked.shape}; skipped {len(bad)} files.")
    out_png = os.path.join(FOLDER, "stacked_median.png")
    plot_image(stacked, title=f"Median stack of {len(fits_list)-len(bad)} files", savepath=out_png)
    out_fits = os.path.join(FOLDER, "reduced_crab_g_master_with_stars.fits")
    fits.writeto(out_fits, stacked, overwrite=True)
    print("Wrote:", out_fits)
