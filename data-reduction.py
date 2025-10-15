import os
import numpy as np
from astropy.io import fits
from tkinter import Tk, filedialog

def open_folder_dialog(title="Select Folder"):
    root = Tk(); root.withdraw()
    folder = filedialog.askdirectory(title=title)
    root.destroy()
    return folder

def open_file_dialog(title="Select File", filetypes=(("FITS files", "*.fits *.fit"),)):
    root = Tk(); root.withdraw()
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return path

def build_gain_table(flat_data):
    medval = np.nanmedian(flat_data)
    print(f"🔧 MasterFlat median value: {medval}")
    if medval == 0:
        raise ValueError("MasterFlat has zero max value; cannot normalize.")
    return flat_data / medval

def reduce_data(raw_folder, masterdark_path, masterflat_path, output_folder):
    print(f"\n📂 Raw folder: {raw_folder}")
    print(f"🌑 MasterDark: {masterdark_path}")
    print(f"🔆 MasterFlat: {masterflat_path}")
    print(f"📤 Output folder: {output_folder}\n")

    # Load master dark
    with fits.open(masterdark_path) as hd:
        md_data = hd[0].data.astype(float)
        md_hdr  = hd[0].header
    print(f"✅ Loaded MasterDark | shape: {md_data.shape}")

    # Load master flat
    with fits.open(masterflat_path) as hf:
        mf_data = hf[0].data.astype(float)
        mf_hdr  = hf[0].header
    print(f"✅ Loaded MasterFlat | shape: {mf_data.shape}")

    # Build gain table
    gain_table = build_gain_table(mf_data)
    print(f"✅ Gain table built | shape: {gain_table.shape}")

    # Ensure output exists
    os.makedirs(output_folder, exist_ok=True)

    # Collect raw files
    raw_files = sorted(f for f in os.listdir(raw_folder) if f.lower().endswith((".fits", ".fit")))
    print(f"🔍 Found {len(raw_files)} raw files to process.\n")

    for fname in raw_files:
        raw_path = os.path.join(raw_folder, fname)
        print(f"📄 Processing: {fname}")

        try:
            with fits.open(raw_path) as hr:
                raw_data = hr[0].data.astype(float)
                raw_hdr  = hr[0].header
        except Exception as e:
            print(f"❌ Failed to open {fname}: {e}")
            continue

        # Shape check
        if raw_data.shape != md_data.shape or raw_data.shape != gain_table.shape:
            print(f"❌ Shape mismatch: raw={raw_data.shape}, dark={md_data.shape}, flat={gain_table.shape}")
            continue

        # Apply corrections
        corrected = (raw_data - md_data) / gain_table

        # Build new header
        new_hdr = raw_hdr.copy()
        new_hdr['AUTHOR']   = "Aysan Hemmati"
        new_hdr['HISTORY']  = "Dark-subtracted and flat-field corrected."
        new_hdr['DARKFILE'] = os.path.basename(masterdark_path)
        new_hdr['FLATFILE'] = os.path.basename(masterflat_path)
        new_hdr['GAINMAX']  = np.nanmax(mf_data)
        

        # Output filename
        base, ext = os.path.splitext(fname)
        outname = f"{base}_dark_and_flat_corrected{ext}"
        outpath = os.path.join(output_folder, outname)

        # Write corrected FITS
        try:
            hdu = fits.PrimaryHDU(corrected, header=new_hdr)
            hdu.writeto(outpath, overwrite=True)
            print(f"✅ Written: {outpath}\n")
        except Exception as e:
            print(f"❌ Failed to write {outname}: {e}\n")

if __name__ == "__main__":
    raw_folder    = open_folder_dialog("Select Folder with Raw FITS Frames")
    masterdark    = open_file_dialog("Select MasterDark FITS")
    masterflat    = open_file_dialog("Select MasterFlat FITS")
    output_folder = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\reduced"

    reduce_data(raw_folder, masterdark, masterflat, output_folder)
    print("🎉 Reduction complete!")