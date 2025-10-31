import os
import numpy as np
from astropy.io import fits
import astroalign as aa
from glob import glob

# 📁 Input and output directories
input_dir = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\rezaei_saba_farideH_2025_10_21\ngc604\r\high"
output_dir = os.path.join(input_dir, "aligned")
os.makedirs(output_dir, exist_ok=True)

# 📂 Load all FITS/fit files
fits_files = sorted(glob(os.path.join(input_dir, "*.fit*")))
print(f"Found {len(fits_files)} FITS files.")

# 📌 Use the first image as reference
ref_data = fits.getdata(fits_files[0])
ref_data = np.ascontiguousarray(ref_data.byteswap().newbyteorder())
aligned_images = [ref_data]

# 🧭 Align all other images
for path in fits_files[1:]:
    try:
        data = fits.getdata(path)
        data = np.ascontiguousarray(data.byteswap().newbyteorder())  # Convert to native byte order

        aligned, _ = aa.register(data, ref_data)
        aligned_images.append(aligned)

        # Save aligned image
        aligned_path = os.path.join(output_dir, f"aligned_{os.path.basename(path)}")
        fits.writeto(aligned_path, aligned.astype(np.float32), overwrite=True)
        print(f"Aligned and saved: {aligned_path}")

    except Exception as e:
        print(f"Failed to align {path}: {e}")

# 📊 Stack aligned images (mean)
stacked = np.mean(aligned_images, axis=0)
stacked_path = os.path.join(output_dir, "NGC604_r.fits")
fits.writeto(stacked_path, stacked.astype(np.float32), overwrite=True)
print(f"Stacked image saved to: {stacked_path}")
