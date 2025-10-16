#!/usr/bin/env python3

import os
import glob
import json
import numpy as np
import astroalign as aa
from astropy.io import fits
from skimage.transform import AffineTransform

# --------------------------------------------------
# USER PATHS
# --------------------------------------------------

REFERENCE_FITS = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\Sep30\Rezaei_30_sep_2025\target3\g\high\1 min\aligned\target3_g_T10C_2025_10_01_2x2_exp00.01.00.000_000001_High_4.fit"
HIGH_FOLDER    = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\Sep30\Rezaei_30_sep_2025\target3\g\high\2 min"
LOW_FOLDER     = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\Sep30\Rezaei_30_sep_2025\target3\g\low\2 min"

OUT_HIGH_ALIGNED = os.path.join(HIGH_FOLDER, "aligned_to_ref")
OUT_LOW_ALIGNED  = os.path.join(LOW_FOLDER, "aligned_to_ref")
TRANSFORM_FILE   = os.path.join(HIGH_FOLDER, "high_to_ref_transforms.json")

# --------------------------------------------------
# UTILITIES
# --------------------------------------------------

def list_fits(folder):
    pats = ("*.fits", "*.fit", "*.fz", "*.fts")
    files = []
    for p in pats:
        files.extend(sorted(glob.glob(os.path.join(folder, p))))
    return files


def read_primary(path):
    """
    Read FITS data and automatically extract the correct plane.
    - Always takes index 0 for consistency.
    - If 3D (2, H, W): extract plane 0.
    """
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data
        hdr = hdul[0].header.copy()

        if data is None:
            raise ValueError(f"No image data in {path}")

        data = np.asarray(data, dtype=float)

        # If multi-plane FITS (2 x 2048 x 2048) → take index 0
        if data.ndim == 3:
            data = data[0]

        if data.ndim != 2:
            raise ValueError(f"{path} → unexpected shape {data.shape}")

        return data, hdr


def write_fits(path, data, hdr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    hdr["HISTORY"] = "Aligned using astroalign (external reference)"
    fits.PrimaryHDU(data, header=hdr).writeto(path, overwrite=True)

# --------------------------------------------------
# STEP 1 — ALIGN HIGH IMAGES TO EXTERNAL REFERENCE
# --------------------------------------------------

def align_high_images_to_reference():
    print(f"\n🛰️ Aligning high images in: {HIGH_FOLDER}")
    print(f"Using external reference FITS: {REFERENCE_FITS}")

    # Load external reference
    ref_img, ref_hdr = read_primary(REFERENCE_FITS)

    files = list_fits(HIGH_FOLDER)
    if not files:
        raise RuntimeError("No FITS files found in high folder.")

    transforms = {}

    for path in files:
        fname = os.path.basename(path)
        print(f"Aligning {fname} → reference ...")
        try:
            img, hdr = read_primary(path)
            transf, _ = aa.find_transform(img, ref_img)
            aligned_img = aa.apply_transform(transf, img, ref_img)
            out_path = os.path.join(OUT_HIGH_ALIGNED, fname)
            write_fits(out_path, aligned_img, hdr)

            transforms[fname] = {
                "rotation": transf.rotation,
                "scale": transf.scale,
                "translation": list(transf.translation),
                "matrix": transf.params.tolist()
            }

        except Exception as e:
            print(f"❌ Failed to align {fname}: {e}")

    # Save transforms for Step 2
    with open(TRANSFORM_FILE, "w") as f:
        json.dump(transforms, f, indent=2)

    print(f"✅ Saved {len(transforms)} transforms → {TRANSFORM_FILE}")
    print(f"✅ High images aligned to external reference → {OUT_HIGH_ALIGNED}")

# --------------------------------------------------
# STEP 2 — APPLY TRANSFORMS TO LOW IMAGES
# --------------------------------------------------

def apply_transforms_to_low():
    print(f"\n⚙️ Applying transforms to low images in: {LOW_FOLDER}")

    if not os.path.exists(TRANSFORM_FILE):
        raise FileNotFoundError("No transform file found! Run Step 1 first.")

    with open(TRANSFORM_FILE, "r") as f:
        transforms = json.load(f)

    # Use external reference for low alignment target as well
    ref_img, ref_hdr = read_primary(REFERENCE_FITS)

    for high_name, tdata in transforms.items():
        low_name = high_name.replace("_High_", "_Low_")
        low_path = os.path.join(LOW_FOLDER, low_name)
        if not os.path.exists(low_path):
            print(f"⚠️ Missing low file for: {low_name}")
            continue

        print(f"Applying transform to: {low_name}")
        try:
            img, hdr = read_primary(low_path)
            M = np.array(tdata["matrix"])
            transf = AffineTransform(matrix=M)
            aligned = aa.apply_transform(transf, img, ref_img)
            out_path = os.path.join(OUT_LOW_ALIGNED, low_name)
            write_fits(out_path, aligned, hdr)
        except Exception as e:
            print(f"❌ Failed {low_name}: {e}")

    print(f"✅ All possible low images aligned → {OUT_LOW_ALIGNED}")

# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    print("\n=== ALIGNMENT TO EXTERNAL REFERENCE PIPELINE ===\n")
    align_high_images_to_reference()
    apply_transforms_to_low()
    print("\n✅ Alignment complete.")
    print("You can run check_alignment(OUT_LOW_ALIGNED) in Python to visually verify.")

# --------------------------------------------------

if __name__ == "__main__":
    main()
