import os
import numpy as np
from astropy.io import fits

# Hardcoded paths
flat_folder   = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\masterframes\Sept 30 masterflat\green flats"
science_path  = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\Sep30\target3\g\high\area95_g_T10C_2025_10_02_2x2_exp00.01.00.000_000001_High_1.fit"
output_folder = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\multi_flat_corrected"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Load science image
with fits.open(science_path) as hs:
    sci_data = hs[0].data.astype(float)
    sci_hdr  = hs[0].header
print(f"✅ Loaded science image | shape: {sci_data.shape}")

# Collect flat files
flat_files = sorted(f for f in os.listdir(flat_folder) if f.lower().endswith((".fits", ".fit")))
print(f"🔍 Found {len(flat_files)} flat frames.\n")

for fname in flat_files:
    flat_path = os.path.join(flat_folder, fname)
    print(f"🔧 Processing flat: {fname}")

    try:
        with fits.open(flat_path) as hf:
            flat_data = hf[0].data.astype(float)
            flat_hdr  = hf[0].header
    except Exception as e:
        print(f"❌ Failed to open flat {fname}: {e}")
        continue

    # Shape check
    if flat_data.shape != sci_data.shape:
        print(f"❌ Shape mismatch: flat={flat_data.shape}, science={sci_data.shape}")
        continue

    # Build gain table
    medval = np.nanmedian(flat_data)
    if medval == 0:
        print(f"❌ Flat {fname} has zero median; skipping.")
        continue
    gain_table = flat_data / medval

    # Apply flat correction
    corrected = sci_data / gain_table

    # Build new header
    new_hdr = sci_hdr.copy()
    new_hdr['AUTHOR']   = "Aysan Hemmati"
    new_hdr['HISTORY']  = f"Flat-field corrected using {fname}"
    new_hdr['FLATFILE'] = fname
    new_hdr['GAINMAX']  = np.nanmax(flat_data)

    # Output filename
    base = os.path.splitext(fname)[0]
    outname = f"galaxy_B_001_flat_corrected_with_{base}.fit"
    outpath = os.path.join(output_folder, outname)

    # Write corrected image
    try:
        hdu = fits.PrimaryHDU(corrected, header=new_hdr)
        hdu.writeto(outpath, overwrite=True)
        print(f"✅ Written: {outpath}\n")
    except Exception as e:
        print(f"❌ Failed to write {outname}: {e}\n")

print("🎉 All flat corrections complete.")