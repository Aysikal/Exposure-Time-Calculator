#!/usr/bin/env python3
import glob
import subprocess
import os
import shutil
import sys

# ─── Configuration ────────────────────────────────────────────────
FOLDER     = r"C:\Users\AYSAN\Desktop\project\INO\gd246\Test folder"
SCALE_LOW  = 0.04    # arcsec/pix (~20% below 0.047)
SCALE_HIGH = 0.06    # arcsec/pix (~20% above 0.047)
# ─────────────────────────────────────────────────────────────────

# 1) Locate solve-field on your PATH
solve_field = shutil.which("solve-field")
if solve_field is None:
    print("✗ ERROR: solve-field not found.")
    print("  • Make sure you’ve activated the Conda env where you ran:")
    print("      conda install -c conda-forge astrometry-net")
    print("  • Or set solve_field = r'C:\\full\\path\\to\\solve-field.exe'")
    sys.exit(1)

BASE_ARGS = [
    solve_field,
    "--overwrite",
    "--no-plots",
    "--scale-units", "arcsecperpix",
    "--scale-low",  str(SCALE_LOW),
    "--scale-high", str(SCALE_HIGH),
    "--new-fits",     # bake WCS into a new .new file
]

# 2) Grab both .fit and .fits
patterns = [os.path.join(FOLDER, "*.fit"), os.path.join(FOLDER, "*.fits")]
fits_files = []
for pat in patterns:
    fits_files.extend(glob.glob(pat))

if not fits_files:
    print(f"✗ No .fit or .fits files found in\n  {FOLDER!r}")
    sys.exit(1)

# 3) Loop & solve
for fits_file in fits_files:
    name = os.path.basename(fits_file)
    print(f"\n→ Solving {name}…")
    cmd = BASE_ARGS + [fits_file]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        err = result.stderr.strip().splitlines()[-1]
        print(f"  ✗ Failed: {err}")
        continue

    new_fits = fits_file.replace(".fit", ".new")
    # If your original was .fits → .new fits
    new_fits = new_fits if os.path.exists(new_fits) else fits_file.replace(".fits", ".new")
    if os.path.exists(new_fits):
        print("  ✓ WCS baked into:", os.path.basename(new_fits))
    else:
        print("  ✗ .new file not found—check solve-field output!")
print("\nDone.")