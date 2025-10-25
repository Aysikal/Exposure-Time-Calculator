import os
import numpy as np
from astropy.io import fits
from tkinter import Tk, filedialog, messagebox

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

def open_multiple_files_dialog(title="Select MasterDark for EXPTIME", filetypes=(("FITS files", "*.fits *.fit"),)):
    root = Tk(); root.withdraw()
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return path

def build_gain_table(flat_data):
    medval = np.nanmedian(flat_data)
    print(f"🔧 MasterFlat median value: {medval}")
    if medval == 0:
        raise ValueError("MasterFlat has zero median value; cannot normalize.")
    return flat_data / medval

def collect_raw_exptimes(raw_folder, raw_files):
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

def reduce_data(raw_folder, masterdark_map, masterflat_path, output_folder):
    print(f"\n📂 Raw folder: {raw_folder}")
    print(f"🔆 MasterFlat: {masterflat_path}")
    print(f"📤 Output folder: {output_folder}\n")

    # Load master flat
    with fits.open(masterflat_path) as hf:
        mf_data = hf[0].data.astype(float)
        mf_hdr  = hf[0].header
    print(f"✅ Loaded MasterFlat | shape: {mf_data.shape}")

    # Build gain table
    gain_table = build_gain_table(mf_data)
    print(f"✅ Gain table built | shape: {gain_table.shape}")

    # Load master darks into memory keyed by exptime
    masterdarks = {}
    for exptime, mpath in masterdark_map.items():
        with fits.open(mpath) as hd:
            md_data = hd[0].data.astype(float)
            md_hdr  = hd[0].header
        masterdarks[exptime] = (md_data, md_hdr, os.path.basename(mpath))
        print(f"✅ Loaded MasterDark for EXPTIME={exptime} | shape: {md_data.shape}")

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

        # Handle 3D FITS (take first plane if needed)
        if raw_data.ndim == 3:
            print(f"ℹ️ {fname} is 3D {raw_data.shape} → using plane [0]")
            raw_data = raw_data[0]

        # Determine exptime for this raw file
        exptime = raw_hdr.get('EXPTIME', raw_hdr.get('EXPOSURE', None))
        if exptime not in masterdarks:
            print(f"❌ No MasterDark provided for EXPTIME={exptime}. Skipping {fname}")
            continue

        md_data, md_hdr, md_basename = masterdarks[exptime]

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
        new_hdr['DARKFILE'] = md_basename
        new_hdr['FLATFILE'] = os.path.basename(masterflat_path)
        new_hdr['GAINMAX']  = np.nanmax(mf_data)
        new_hdr['EXPTIME']  = exptime

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
    masterflat    = open_file_dialog("Select MasterFlat FITS")
    output_folder = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\data 2\Rezaei_Altafi_10_07_2025\standard_star\94 B2\r\high\keep\reduced"

    # Gather raw file list first
    raw_files = sorted(f for f in os.listdir(raw_folder) if f.lower().endswith((".fits", ".fit")))
    if not raw_files:
        raise SystemExit("No raw FITS files found in the selected folder.")

    # Inspect headers to find unique EXPTIME values
    exptime_map = collect_raw_exptimes(raw_folder, raw_files)
    unique_exptimes = list(exptime_map.keys())
    print(f"🔎 Found EXPTIME values: {unique_exptimes}\n")

    # If multiple exptimes, prompt once per unique value for a matching MasterDark
    masterdark_map = {}
    if len(unique_exptimes) == 1:
        md_path = open_file_dialog("Select MasterDark FITS for the single EXPTIME")
        masterdark_map[unique_exptimes[0]] = md_path
    else:
        # Sort for deterministic ordering
        for exptime in sorted(unique_exptimes, key=lambda x: (x is None, x)):
            title = f"Select MasterDark FITS for EXPTIME={exptime}"
            md_path = open_file_dialog(title)
            masterdark_map[exptime] = md_path

    reduce_data(raw_folder, masterdark_map, masterflat, output_folder)
    print("🎉 Reduction complete!")
