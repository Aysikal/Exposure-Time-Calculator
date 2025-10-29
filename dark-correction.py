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

def collect_raw_exptimes(raw_folder, raw_files):
    """Collect all unique exposure times from raw FITS headers."""
    exptimes = {}
    for fname in raw_files:
        path = os.path.join(raw_folder, fname)
        try:
            with fits.open(path) as h:
                hdr = h[0].header
                exptime = hdr.get('EXPTIME', hdr.get('EXPOSURE', None))
                if exptime is None:
                    print(f"⚠️ {fname} missing EXPTIME; treating as EXPTIME=None")
                exptimes.setdefault(exptime, []).append(fname)
        except Exception as e:
            print(f"❌ Failed to read header for {fname}: {e}")
    return exptimes

def dark_subtract(raw_folder, masterdark_map, output_folder):
    """Apply dark subtraction only."""
    print(f"\n📂 Raw folder: {raw_folder}")
    print(f"📤 Output folder: {output_folder}\n")

    # Load all master darks into memory keyed by exptime
    masterdarks = {}
    for exptime, mpath in masterdark_map.items():
        with fits.open(mpath) as hd:
            md_data = hd[0].data.astype(float)
            md_hdr  = hd[0].header
        masterdarks[exptime] = (md_data, md_hdr, os.path.basename(mpath))
        print(f"✅ Loaded MasterDark for EXPTIME={exptime} | shape: {md_data.shape}")

    os.makedirs(output_folder, exist_ok=True)

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

        if raw_data.ndim == 3:
            print(f"ℹ️ {fname} is 3D {raw_data.shape} → using plane [0]")
            raw_data = raw_data[0]

        exptime = raw_hdr.get('EXPTIME', raw_hdr.get('EXPOSURE', None))
        if exptime not in masterdarks:
            print(f"❌ No MasterDark provided for EXPTIME={exptime}. Skipping {fname}")
            continue

        md_data, md_hdr, md_basename = masterdarks[exptime]

        if raw_data.shape != md_data.shape:
            print(f"❌ Shape mismatch: raw={raw_data.shape}, dark={md_data.shape}")
            continue

        # 🔧 Apply dark subtraction only
        corrected = raw_data - md_data

        # 🧾 Build updated header
        new_hdr = raw_hdr.copy()
        new_hdr['AUTHOR']   = "Aysan Hemmati"
        new_hdr['HISTORY']  = "Dark-subtracted only."
        new_hdr['DARKFILE'] = md_basename
        new_hdr['EXPTIME']  = exptime

        base, ext = os.path.splitext(fname)
        outname = f"{base}_dark_corrected{ext}"
        outpath = os.path.join(output_folder, outname)

        try:
            fits.PrimaryHDU(corrected, header=new_hdr).writeto(outpath, overwrite=True)
            print(f"✅ Written: {outpath}\n")
        except Exception as e:
            print(f"❌ Failed to write {outname}: {e}\n")

if __name__ == "__main__":
    raw_folder = open_folder_dialog("Select Folder with Raw FITS Frames")
    output_folder = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\dark_corrected"

    # Gather raw file list
    raw_files = sorted(f for f in os.listdir(raw_folder) if f.lower().endswith((".fits", ".fit")))
    if not raw_files:
        raise SystemExit("No raw FITS files found in the selected folder.")

    # Inspect headers for unique EXPTIME values
    exptime_map = collect_raw_exptimes(raw_folder, raw_files)
    unique_exptimes = list(exptime_map.keys())
    print(f"🔎 Found EXPTIME values: {unique_exptimes}\n")

    # Prompt for a master dark for each unique exposure
    masterdark_map = {}
    if len(unique_exptimes) == 1:
        md_path = open_file_dialog("Select MasterDark FITS for this EXPTIME")
        masterdark_map[unique_exptimes[0]] = md_path
    else:
        for exptime in sorted(unique_exptimes, key=lambda x: (x is None, x)):
            title = f"Select MasterDark FITS for EXPTIME={exptime}"
            md_path = open_file_dialog(title)
            masterdark_map[exptime] = md_path

    dark_subtract(raw_folder, masterdark_map, output_folder)
    print("🎉 Dark correction complete!")
