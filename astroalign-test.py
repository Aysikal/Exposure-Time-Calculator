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

HIGH_FOLDER = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\Sep30\Rezaei_30_sep_2025\target3\u\high"
LOW_FOLDER  = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\Sep30\Rezaei_30_sep_2025\target3\u\low"
OUT_HIGH_ALIGNED = os.path.join(HIGH_FOLDER, "aligned")
OUT_LOW_ALIGNED  = os.path.join(LOW_FOLDER, "aligned")
TRANSFORM_FILE   = os.path.join(HIGH_FOLDER, "high_transforms.json")

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
    Read FITS data and automatically extract only the correct plane.
    - If 2 planes are found (shape = [2, H, W]):
        * '_High_' in filename → first plane (index 0)
        * '_Low_'  in filename → second plane (index 1)
    - Otherwise returns 2D image directly.
    """
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data
        hdr = hdul[0].header.copy()

        if data is None:
            raise ValueError(f"No image data in {path}")

        data = np.asarray(data, dtype=float)

        
        if data.ndim == 3 and data.shape[0] == 2:
            fname = os.path.basename(path).lower()
            if "_high_" in fname:
                data = data[0]  # keep first plane
            elif "_low_" in fname:
                data = data[0]  # keep second plane
            else:
                print(f"⚠️ {fname}: 2 planes detected but no 'High/Low' tag — using first plane.")
                data = data[0]

        # Final sanity check
        if data.ndim != 2:
            raise ValueError(f"{path} → unexpected shape {data.shape}")

        return data, hdr


def write_fits(path, data, hdr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    hdr["HISTORY"] = "Aligned using astroalign or precomputed transform"
    fits.PrimaryHDU(data, header=hdr).writeto(path, overwrite=True)


# --------------------------------------------------
# STEP 1 — ALIGN HIGH IMAGES
# --------------------------------------------------

def align_high_images():
    print(f"\n🛰️ Aligning high images in: {HIGH_FOLDER}")
    files = list_fits(HIGH_FOLDER)
    if len(files) < 2:
        raise RuntimeError("Need at least 2 FITS files in high folder.")

    ref_path = files[0]
    ref_img, ref_hdr = read_primary(ref_path)
    transforms = {}

    ref_out = os.path.join(OUT_HIGH_ALIGNED, os.path.basename(ref_path))
    write_fits(ref_out, ref_img, ref_hdr)
    print(f"Reference frame: {os.path.basename(ref_path)}")

    for path in files[1:]:
        fname = os.path.basename(path)
        print(f"Aligning {fname} ...")
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

    with open(TRANSFORM_FILE, "w") as f:
        json.dump(transforms, f, indent=2)

    print(f"✅ Saved {len(transforms)} transforms → {TRANSFORM_FILE}")
    print(f"✅ Aligned high images saved to: {OUT_HIGH_ALIGNED}")


# --------------------------------------------------
# STEP 2 — APPLY TRANSFORMS TO LOW IMAGES
# --------------------------------------------------

def apply_transforms_to_low():
    print(f"\n⚙️ Applying transforms to low images in: {LOW_FOLDER}")

    if not os.path.exists(TRANSFORM_FILE):
        raise FileNotFoundError("No transform file found! Run Step 1 first.")

    with open(TRANSFORM_FILE, "r") as f:
        transforms = json.load(f)

    ref_name = list(transforms.keys())[0]
    ref_low_name = ref_name.replace("_High_", "_Low_")
    ref_low_path = os.path.join(LOW_FOLDER, ref_low_name)
    if not os.path.exists(ref_low_path):
        raise FileNotFoundError(f"Reference low image not found: {ref_low_path}")

    ref_img, ref_hdr = read_primary(ref_low_path)
    write_fits(os.path.join(OUT_LOW_ALIGNED, ref_low_name), ref_img, ref_hdr)
    print(f"Reference low frame: {ref_low_name}")

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
    print("\n=== HIGH → LOW ALIGNMENT PIPELINE ===\n")
    align_high_images()
    apply_transforms_to_low()
    print("\n✅ Alignment complete.")
    print("You can run check_alignment(OUT_LOW_ALIGNED) in Python to visually verify.")


if __name__ == "__main__":
    main()
