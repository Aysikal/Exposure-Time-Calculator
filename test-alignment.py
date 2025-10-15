import os
import glob
import numpy as np
from astropy.io import fits
import astroalign as aa

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
INPUT_DIR = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\reduced\sept 30 area 95 g low\keep"           # folder with your raw FITS
ALIGNED_DIR = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\reduced\sept 30 area 95 g low\keep\aligned"      # output folder for aligned files
STACKED_PATH = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\reduced\sept 30 area 95 g low\keep\stacked.fits"

os.makedirs(ALIGNED_DIR, exist_ok=True)

# ----------------------------------------------------
# UTILITY
# ----------------------------------------------------
def load_fits_le(path):
    """Load FITS and ensure little-endian array (for numpy compatibility)."""
    data = fits.getdata(path)
    header = fits.getheader(path)

    # Convert from big-endian to little-endian if necessary
    if data.dtype.byteorder == ">" or (data.dtype.byteorder == "=" and np.little_endian is False):
        data = data.byteswap().newbyteorder()

    return data.astype(np.float32), header


# ----------------------------------------------------
# LOAD FILES
# ----------------------------------------------------
files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.fit*")))
if len(files) < 2:
    raise SystemExit("Need at least two FITS files to align!")

# Choose middle frame as reference
ref_path = files[len(files)//2]
ref_data, ref_header = load_fits_le(ref_path)

print(f"Reference frame: {os.path.basename(ref_path)}")

# ----------------------------------------------------
# ALIGN ALL FRAMES
# ----------------------------------------------------
aligned_images = [ref_data]

for fpath in files:
    fname = os.path.basename(fpath)
    if fpath == ref_path:
        continue

    try:
        data, _ = load_fits_le(fpath)
        aligned, footprint = aa.register(data, ref_data)
        aligned_images.append(aligned)
        out_path = os.path.join(ALIGNED_DIR, f"aligned_{fname}")
        fits.writeto(out_path, aligned.astype(np.float32), ref_header, overwrite=True)
        print(f"✅ {fname} aligned successfully")

    except aa.MaxIterError:
        print(f"❌ {fname} failed to align (max iterations)")
    except Exception as e:
        print(f"❌ {fname} failed: {e}")

# ----------------------------------------------------
# STACK
# ----------------------------------------------------
if aligned_images:
    stack = np.nanmedian(np.stack(aligned_images, axis=0), axis=0)
    fits.writeto(STACKED_PATH, stack.astype(np.float32), ref_header, overwrite=True)
    print(f"\n📦 Stacked FITS saved to: {STACKED_PATH}")
else:
    print("⚠️ No frames aligned successfully — stack not created.")
