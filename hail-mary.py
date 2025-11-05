import os
import numpy as np
from astropy.io import fits
import astroalign as aa
from glob import glob

# 📁 Input and output directories
input_dir = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\Sep30\Rezaei_30_sep_2025\target3\r\high\keep\hot pixels removed"
output_dir = os.path.join(input_dir, "aligned")
os.makedirs(output_dir, exist_ok=True)

# 📂 Load all FITS/fit files
fits_files = sorted(glob(os.path.join(input_dir, "*.fit*")))
print(f"Found {len(fits_files)} FITS files.")

# 📌 Use the first image as reference
ref_hdul = fits.open(fits_files[0])
ref_data = np.ascontiguousarray(ref_hdul[0].data.byteswap().newbyteorder())
ref_header = ref_hdul[0].header
ref_hdul.close()

aligned_images = [ref_data]

# 🧭 Align all other images
for path in fits_files[1:]:
    try:
        hdul = fits.open(path)
        data = np.ascontiguousarray(hdul[0].data.byteswap().newbyteorder())
        header = hdul[0].header
        hdul.close()

        aligned, _ = aa.register(data, ref_data, max_control_points=50, detection_sigma=4)
        aligned_images.append(aligned)

        # Save aligned image with original header
        aligned_path = os.path.join(output_dir, f"aligned_{os.path.basename(path)}")
        fits.writeto(aligned_path, aligned.astype(np.float32), header=header, overwrite=True)
        print(f"Aligned and saved: {aligned_path}")

    except Exception as e:
        print(f"Failed to align {path}: {e}")

# 📊 Stack aligned images (median and sum) using reference header
stacked = np.median(aligned_images, axis=0)
summed = np.sum(aligned_images, axis=0)

stacked_path = os.path.join(output_dir, "Area95_r_sept30_median.fits")
sum_path = os.path.join(output_dir, "AREA995_I_sept30_sum.fits")

fits.writeto(stacked_path, stacked.astype(np.float32), header=ref_header, overwrite=True)
fits.writeto(sum_path, summed.astype(np.float32), header=ref_header, overwrite=True)

print(f"Stacked images saved to:\n- Median: {stacked_path}\n- Sum: {sum_path}")
