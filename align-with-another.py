#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced FITS Image Alignment Pipeline
--------------------------------------
This script performs a two-stage alignment process for astronomical data:
    1. Aligns all "High" exposure images to an external reference frame.
    2. Applies those computed affine transformations to corresponding "Low" exposure images.
    
The alignment is performed using Astroalign, ensuring pixel-perfect registration
while maintaining metadata consistency in the FITS headers.

Author: Aysan Alignment Project (Rezaei_30_Sep_2025)
Last Updated: 2025-10-16
"""

import os
import glob
import json
import numpy as np
import astroalign as aa
from astropy.io import fits
from skimage.transform import AffineTransform

# =============================================================================
# CONFIGURATION SECTION
# =============================================================================

REFERENCE_FITS = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\Sep30\Rezaei_30_sep_2025\target3\g\high\1 min\aligned\target3_g_T10C_2025_10_01_2x2_exp00.01.00.000_000001_High_4.fit"

HIGH_FOLDER    = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\Sep30\Rezaei_30_sep_2025\target3\g\high\2 min"
LOW_FOLDER     = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\Sep30\Rezaei_30_sep_2025\target3\g\low\2 min"

OUT_HIGH_ALIGNED = os.path.join(HIGH_FOLDER, "aligned_to_ref")
OUT_LOW_ALIGNED  = os.path.join(LOW_FOLDER, "aligned_to_ref")

TRANSFORM_FILE   = os.path.join(HIGH_FOLDER, "high_to_ref_transforms.json")

# =============================================================================
# FILE UTILITIES
# =============================================================================

def list_fits(folder):
    """
    List all FITS-compatible files in the given folder.

    Parameters
    ----------
    folder : str
        Directory containing FITS images.

    Returns
    -------
    files : list of str
        Sorted list of matching FITS paths.
    """
    patterns = ("*.fits", "*.fit", "*.fz", "*.fts")
    fits_files = []
    for pattern in patterns:
        candidate_files = sorted(glob.glob(os.path.join(folder, pattern)))
        fits_files.extend(candidate_files)
    return fits_files


def read_primary(path):
    """
    Read a FITS file, ensuring extraction of the primary image plane (index 0).

    If the FITS contains multiple planes, the first is taken consistently.

    Parameters
    ----------
    path : str
        Full path to FITS file.

    Returns
    -------
    data : np.ndarray
        2D array of image pixel values.
    hdr : fits.Header
        FITS header associated with the primary HDU.
    """
    with fits.open(path, memmap=False) as hdul:
        hdr = hdul[0].header.copy()
        raw_data = hdul[0].data

        if raw_data is None:
            raise ValueError(f"[read_primary] No image data found in: {path}")

        data = np.asarray(raw_data, dtype=float)

        # Handle possible multi-plane FITS data
        if data.ndim == 3:
            if data.shape[0] > 1:
                print(f"[info] Multi-plane FITS detected ({data.shape}); extracting plane 0.")
            data = data[0]

        # Validate final shape
        if data.ndim != 2:
            raise ValueError(f"[read_primary] Unexpected shape {data.shape} for {path}")

        return data, hdr


def write_fits(path, data, hdr):
    """
    Write a new FITS file preserving metadata and adding a provenance tag.

    Parameters
    ----------
    path : str
        Output FITS path.
    data : np.ndarray
        2D image data to save.
    hdr : fits.Header
        Original header to attach.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Record processing history
    hdr["HISTORY"] = "Aligned using astroalign (external reference)"
    hdu = fits.PrimaryHDU(data=data, header=hdr)

    hdu.writeto(path, overwrite=True)
    print(f"[write_fits] Saved aligned FITS → {path}")

# =============================================================================
# STAGE 1: ALIGN HIGH-EXPOSURE IMAGES TO REFERENCE
# =============================================================================

def align_high_images_to_reference():
    """
    Align each high-exposure FITS image in HIGH_FOLDER to the external reference image.
    Generates a JSON transform file used later for low-exposure alignment.
    """
    print("\n──────────────────────────────────────────────────────────")
    print(f"🛰️  ALIGNING HIGH-EXPOSURE FRAMES\n→ Folder: {HIGH_FOLDER}")
    print(f"→ Reference: {REFERENCE_FITS}")
    print("──────────────────────────────────────────────────────────")

    # Load external reference
    ref_img, ref_hdr = read_primary(REFERENCE_FITS)
    file_list = list_fits(HIGH_FOLDER)

    if not file_list:
        raise RuntimeError(f"[Error] No FITS files found in {HIGH_FOLDER}")

    transforms = {}

    for idx, path in enumerate(file_list, start=1):
        fname = os.path.basename(path)
        print(f"\n[{idx}/{len(file_list)}] Processing {fname} → reference ...")

        try:
            img, hdr = read_primary(path)

            # Attempt transformation
            transf, _ = aa.find_transform(img, ref_img)
            aligned_img = aa.apply_transform(transf, img, ref_img)

            out_path = os.path.join(OUT_HIGH_ALIGNED, fname)
            write_fits(out_path, aligned_img, hdr)

            # Store transformation metadata
            transforms[fname] = {
                "rotation": transf.rotation,
                "scale": transf.scale,
                "translation": list(transf.translation),
                "matrix": transf.params.tolist()
            }

            print(f"✅ Success: {fname} aligned.")
        except Exception as e:
            print(f"❌ Alignment failed for {fname}: {e}")

    # Save transformation parameters
    with open(TRANSFORM_FILE, "w") as f:
        json.dump(transforms, f, indent=2)

    print("\n──────────────────────────────────────────────────────────")
    print(f"✅ Saved {len(transforms)} valid transforms → {TRANSFORM_FILE}")
    print(f"✅ Aligned outputs → {OUT_HIGH_ALIGNED}")
    print("──────────────────────────────────────────────────────────")


# =============================================================================
# STAGE 2: APPLY TRANSFORMS TO LOW-EXPOSURE IMAGES
# =============================================================================

def apply_transforms_to_low():
    """
    Applies previously derived transforms (from high-exposure images)
    to their corresponding low-exposure pairs.
    """
    print("\n──────────────────────────────────────────────────────────")
    print("⚙️  APPLYING TRANSFORMS TO LOW-EXPOSURE FRAMES")
    print("──────────────────────────────────────────────────────────")

    if not os.path.exists(TRANSFORM_FILE):
        raise FileNotFoundError("Missing transform JSON file — run Stage 1 first!")

    with open(TRANSFORM_FILE, "r") as f:
        transforms = json.load(f)

    # Reference image reused as final alignment coordinate base
    ref_img, ref_hdr = read_primary(REFERENCE_FITS)

    total = len(transforms)
    success_count = 0

    for i, (high_name, tdata) in enumerate(transforms.items(), start=1):
        low_name = high_name.replace("_High_", "_Low_")
        low_path = os.path.join(LOW_FOLDER, low_name)

        print(f"\n[{i}/{total}] Applying transform → {low_name}")

        if not os.path.exists(low_path):
            print(f"⚠️  Skipping: {low_name} not found.")
            continue

        try:
            img, hdr = read_primary(low_path)
            matrix = np.array(tdata["matrix"])
            transf = AffineTransform(matrix=matrix)

            aligned_low = aa.apply_transform(transf, img, ref_img)
            out_path = os.path.join(OUT_LOW_ALIGNED, low_name)
            write_fits(out_path, aligned_low, hdr)
            success_count += 1
        except Exception as e:
            print(f"❌ Failed to align {low_name}: {e}")

    print("\n──────────────────────────────────────────────────────────")
    print(f"✅ Alignment applied to {success_count}/{total} low-exposure frames")
    print(f"✅ Results saved in: {OUT_LOW_ALIGNED}")
    print("──────────────────────────────────────────────────────────")


# =============================================================================
# MAIN EXECUTION LOGIC
# =============================================================================

def main():
    """
    Main entry point for the alignment pipeline.
    Executes both high and low alignment stages sequentially.
    """
    print("\n==========================================================")
    print("=== EXTERNAL REFERENCE ALIGNMENT PIPELINE INITIATED ===")
    print("==========================================================\n")

    align_high_images_to_reference()
    apply_transforms_to_low()

    print("\n==========================================================")
    print("✅ Pipeline completed successfully.")
    print("📈 Use `check_alignment(OUT_LOW_ALIGNED)` to verify visually.")
    print("==========================================================\n")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
