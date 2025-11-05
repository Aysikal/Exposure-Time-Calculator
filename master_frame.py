import os
import re
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from astropy.io import fits
from astropy.stats import sigma_clip
from tkinter import Tk, filedialog, Button, Label
import reza
from scipy.ndimage import median_filter, generic_filter

# --- Hardcoded Mode Tag ---
MODE_TAG = "High"

# --- Regex for extracting color token from flat filenames ---
COLOR_PATTERN = re.compile(r'_(u|g|r|i|clear)_', re.IGNORECASE)

# --- Paths ---
SPREADSHEET_DIR = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\spreadsheets"
MASTER_OUTPUT_DIR = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\masterframes\no hot pixels masterflats"
PLOT_DIR = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\masterframes\no hot pixels masterflats"
# --- Utilities ---
def open_folder_dialog(title="Select Folder"):
    root = Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory(title=title)
    root.destroy()
    return folder_selected

def extract_header_info(header):
    exp_raw = header.get('EXPTIME', None)
    date_obs = header.get('DATE-OBS', 'unknown-date')
    date_obs = date_obs.split('T')[0] if isinstance(date_obs, str) else 'unknown-date'
    xbin = header.get('XBINNING', header.get('BINX', 1))
    ybin = header.get('YBINNING', header.get('BINY', 1))
    try:
        exp_sec = float(exp_raw) if exp_raw is not None else None
    except Exception:
        exp_sec = None
    binning_str = f"bin{xbin}x{ybin}"
    return exp_sec, date_obs, binning_str

def extract_color_from_filename(fname):
    m = COLOR_PATTERN.search(fname)
    return m.group(1).lower() if m else 'unknown'

def log_statistics_to_csv(frame_type, stats_dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{frame_type}.csv")
    file_exists = os.path.isfile(csv_path)
    fieldnames = ["Filename", "FrameType", "ExposureTime", "Date", "Binning",
                  "Mode", "Color", "Mean", "Median", "StdDev", "Min", "Max", "NFrames", "rms"]

    with open(csv_path, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        row = {k: stats_dict.get(k, "") for k in fieldnames}
        writer.writerow(row)

def rms(data):
    data = np.asarray(data)
    return np.sqrt(np.mean(np.square(data)))

def compute_and_print_stats(name, arr):
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        print(f"{name} has no finite pixels to compute stats.")
        return None
    stats = reza.statist(valid)  # expects sequence-like [min, max, mean, std]
    print(f"{name} Stats:")
    print(f"min = {stats[0]:.2f}   max = {stats[1]:.2f}")
    print(f"mean = {stats[2]:.4f}   std = {stats[3]:.4f}")
    return stats, valid

# --- Replacement utilities for clipped pixels ---
def replace_nans_with_local_median(img, primary_size=3, fallback_size=20, mode='mirror'):
    """
    img: 2D array with NaNs marking invalid pixels
    primary_size: small window size (odd)
    fallback_size: larger window size (odd)
    returns: filled array, primary_fill_mask, fallback_fill_mask
    """
    def nanmedian_func(values):
        return np.nanmedian(values)

    # primary pass (e.g., 3x3)
    primary_filtered = generic_filter(img, nanmedian_func, size=primary_size, mode=mode)
    primary_fill_mask = np.isnan(img) & ~np.isnan(primary_filtered)

    # fallback pass for remaining NaNs (e.g., 20x20)
    still_nan_mask = np.isnan(img) & np.isnan(primary_filtered)
    fallback_filtered = generic_filter(img, nanmedian_func, size=fallback_size, mode=mode)
    fallback_fill_mask = still_nan_mask & ~np.isnan(fallback_filtered)

    # compose final filled image
    filled = img.copy()
    filled[primary_fill_mask] = primary_filtered[primary_fill_mask]
    filled[fallback_fill_mask] = fallback_filtered[fallback_fill_mask]

    return filled, primary_fill_mask, fallback_fill_mask

# --- Core processing ---
def create_master_frame(folder_path, output_path, frame_type="dark"):
    files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.fit', '.fits'))])
    if not files:
        print(f"No FITS files found in {folder_path}")
        return

    exp_times, date_obs_list, binning_list = [], [], []
    selected_files, colors, per_file_errors = [], [], []

    for fname in files:
        if MODE_TAG.upper() not in fname.upper():
            continue
        fpath = os.path.join(folder_path, fname)
        try:
            with fits.open(fpath) as hdul:
                header = hdul[0].header
                exp, date_obs, binning = extract_header_info(header)
                exp_times.append(exp * 1e-5 if exp is not None else None)
                date_obs_list.append(date_obs)
                binning_list.append(binning)
                selected_files.append(fname)
                colors.append(extract_color_from_filename(fname))
        except Exception as e:
            per_file_errors.append((fname, str(e)))
            print(f"⚠️ Skipping {fname}: {e}")
            continue

    if not selected_files:
        print(f"❌ No files found with mode tag '{MODE_TAG}' in {frame_type} folder.")
        return

    valid_exp_times = [e for e in exp_times if e is not None]
    if valid_exp_times and len(set(valid_exp_times)) > 1:
        print(f"❌ Exposure times are not uniform in {frame_type} folder.")
        return

    mode_str = MODE_TAG
    exp_str = f"{valid_exp_times[0]:.5f}s" if valid_exp_times else "unknownExp"
    date_str = date_obs_list[0] if date_obs_list else "unknown-date"
    bin_str = binning_list[0] if binning_list else "bin1x1"
    color_for_master = Counter(colors).most_common(1)[0][0] if frame_type.lower() == 'flat' and colors else ''
    color_tag = f"_{color_for_master}" if color_for_master else ""
    base_name = f"master{frame_type}{color_tag}_{exp_str}_{date_str}_{bin_str}_{mode_str}_clipped_and_replaced"
    output_file = os.path.join(output_path, base_name + ".fits")
    plot_file = os.path.join(PLOT_DIR, base_name + ".png")

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # build stack
    first_data_path = os.path.join(folder_path, selected_files[0])
    try:
        shape = fits.getdata(first_data_path).shape
    except Exception as e:
        print(f"❌ Unable to read data shape from {selected_files[0]}: {e}")
        return

    stack = np.zeros(shape, dtype=float)
    for fname in selected_files:
        fpath = os.path.join(folder_path, fname)
        try:
            with fits.open(fpath) as hdul:
                data = hdul[0].data.astype(float)
                stack += data
        except Exception as e:
            print(f"⚠️ Skipping {fname} during stacking: {e}")
            continue

    master = stack / len(selected_files)

    if frame_type.lower() == 'flat':
        print("Applying median filter to suppress stars...")
        master = median_filter(master, size=50, mode="reflect")

    # sigma clip
    clipped = sigma_clip(master, sigma=4, cenfunc='median')
    clipped_nan = clipped.filled(np.nan)

    # replace NaNs: 3x3 then 20x20 fallback
    filled, primary_mask, fallback_mask = replace_nans_with_local_median(clipped_nan, primary_size=3, fallback_size=20, mode='mirror')

    stats_res = compute_and_print_stats("Final clipped and replaced", filled)
    if stats_res is None:
        return
    stats_vals, valid_data = stats_res

    # write final FITS
    hdu = fits.PrimaryHDU(filled)
    hdr = hdu.header
    hdr['NFRAMES'] = len(selected_files)
    hdr['EXPTIME'] = valid_exp_times[0] if valid_exp_times else None
    hdr['BINNING'] = bin_str
    hdr['DATE-MST'] = date_str
    hdr['MODETAG'] = mode_str
    hdr['NREPL3'] = int(np.sum(primary_mask))
    hdr['NREPL20'] = int(np.sum(fallback_mask))
    if frame_type.lower() == 'flat':
        hdr['FILTER'] = color_for_master.upper() if color_for_master else 'UNKNOWN'
    hdr.add_history("Sigma clipped with sigma=4 and replaced NaNs using 3x3 and 20x20 median fallback.")
    hdu.writeto(output_file, overwrite=True)
    print(f"✅ Final FITS saved to: {output_file}")

    # save plot
    plt.figure(figsize=(12, 6), dpi=300)
    plt.subplot(1, 2, 1)
    plt.hist(valid_data, bins=50, edgecolor='black')
    plt.title("Clipped and Replaced Histogram", fontsize=12)
    plt.xlabel("Pixel value", fontsize=10)
    plt.ylabel("Count", fontsize=10)
    plt.subplot(1, 2, 2)
    plt.imshow(filled, cmap='gray', origin='lower', interpolation='none')
    plt.title("Clipped and Replaced Image", fontsize=12)
    plt.colorbar(shrink=0.8, label="Pixel value")
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📁 Plot saved to: {plot_file}")

# --- GUI launcher ---
def launch_gui():
    def on_select(frame_type):
        root.destroy()
        folder_title = f"Select Folder with {frame_type.capitalize()} Frames"
        selected_folder = open_folder_dialog(folder_title)
        if not selected_folder:
            print("No folder selected, aborting.")
            return
        os.makedirs(MASTER_OUTPUT_DIR, exist_ok=True)
        create_master_frame(selected_folder, MASTER_OUTPUT_DIR, frame_type=frame_type)

    root = Tk()
    root.title("Choose Master Frame Type")
    root.geometry("320x170")
    Label(root, text="Select the type of master frame to generate:", pady=10).pack()
    Button(root, text="Masterdark", width=24, command=lambda: on_select("dark")).pack(pady=6)
    Button(root, text="Masterflat", width=24, command=lambda: on_select("flat")).pack(pady=6)
    Button(root, text="Masterbias", width=24, command=lambda: on_select("bias")).pack(pady=6)
    root.mainloop()

# --- Run GUI ---
if __name__ == "__main__":
    launch_gui()

    # --- Completion Sound ---
    try:
        import winsound
        winsound.Beep(1000, 500)  # Frequency: 1000 Hz, Duration: 500 ms
    except ImportError:
        print('\a')  # Fallback for non-Windows systems
