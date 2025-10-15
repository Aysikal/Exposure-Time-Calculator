import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import ZScaleInterval
import astropy.units as u

# Load the FITS image (with WCS info)
fits_file = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Astrometry\new-image.fits"
with fits.open(fits_file) as hdul:
    image_data = hdul[0].data
    wcs = WCS(hdul[0].header)

# Apply ZScale normalization
zscale = ZScaleInterval()
vmin, vmax = zscale.get_limits(image_data)

# Load detected stars (pixel coordinates)
axy_file = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Astrometry\axy.fits"
with fits.open(axy_file) as stars_hdul:
    stars_data = stars_hdul[1].data
    x_coords = stars_data['X']
    y_coords = stars_data['Y']

# Load reference stars (RA/Dec coordinates)
rdls_file = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Astrometry\rdls.fits"
with fits.open(rdls_file) as ref_hdul:
    ref_data = ref_hdul[1].data
    ra_ref = ref_data['RA']
    dec_ref = ref_data['DEC']
    x_ref, y_ref = wcs.wcs_world2pix(ra_ref, dec_ref, 1)

# Create interactive plot
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': wcs})
ax.imshow(image_data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)

# Overlay detected stars
ax.scatter(x_coords, y_coords, s=15, edgecolor='red', facecolor='none', marker='o', label="Detected Stars")

# Overlay reference stars
ax.scatter(x_ref, y_ref, s=20, edgecolor='lime', facecolor='none', marker='s', label="Reference Stars")

# Configure RA/Dec grid
ax.coords.grid(True, color='white', linestyle='dotted', alpha=0.6)
ax.coords[0].set_ticks(spacing=2 * u.arcmin, color='white')
ax.coords[1].set_ticks(spacing=2 * u.arcmin, color='white')
ax.coords[0].set_axislabel("Right Ascension")
ax.coords[1].set_axislabel("Declination")

# Interactive coordinate display
def display_coordinates(event):
    if event.xdata is not None and event.ydata is not None:
        ra, dec = wcs.wcs_pix2world(event.xdata, event.ydata, 1)
        

fig.canvas.mpl_connect("motion_notify_event", display_coordinates)

plt.title("Detected & Reference Stars in Area 95")
plt.legend()
plt.show()

"""
stack_with_hints.py
Batch WCS solve (nova.astrometry.net) + reproject to a common reference grid.
Windows-friendly, robust: retries + cropped fallback + resume support.

Requirements:
pip install astroquery astropy reproject
Get an API key at https://nova.astrometry.net (My Profile -> API Key)
"""

import os
from glob import glob
from time import sleep
from astroquery.astrometry_net import AstrometryNet
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp

# === USER CONFIG ===
fits_dir = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\reduced\October 1st, area 95 green high\keep"
output_dir = os.path.join(fits_dir, "aligned")
os.makedirs(output_dir, exist_ok=True)

API_KEY = "pakymwxyricoavhw"   # <-- put your nova.astrometry.net API key here

# PIXEL_SCALE: set this to your camera/telescope pixel scale in arcsec/pixel.
# If you don't know it, I'll compute it for you if you provide pixel size (um) & focal length (mm).
PIXEL_SCALE = 1.8 * 0.047  # arcsec/pixel (adjust to your setup)

# Hints from you (parsed below)
RA_STR = "03:53:21"    # hours:minutes:seconds
DEC_STR = "-00:00:20"  # degrees:minutes:seconds (leading sign allowed)
RADIUS_DEG = 0.12       # search radius around RA/DEC in degrees (adjust if needed)

# solver parameters
SCALE_MARGIN_LOW = 0.9  # will use PIXEL_SCALE * SCALE_MARGIN_LOW as lower bound
SCALE_MARGIN_HIGH = 1.1 # upper bound multiplier
MAX_RETRIES = 3
RETRY_WAIT = 5  # seconds between retries

# file extensions to find
extensions = ("*.fit", "*.fits")


# === Utilities ===
def hms_to_deg(hms: str) -> float:
    """Convert 'HH:MM:SS' (string) to degrees (float)."""
    parts = [float(p) for p in hms.strip().split(':')]
    if len(parts) != 3:
        raise ValueError("RA must be 'HH:MM:SS'")
    h, m, s = parts
    return 15.0 * (h + m / 60.0 + s / 3600.0)  # 1h = 15 deg

def dms_to_deg(dms: str) -> float:
    """Convert '[+/-]DD:MM:SS' to degrees (float)."""
    sign = 1
    s = dms.strip()
    if s.startswith('-'):
        sign = -1
        s = s[1:]
    elif s.startswith('+'):
        s = s[1:]
    parts = [float(p) for p in s.split(':')]
    if len(parts) != 3:
        raise ValueError("DEC must be 'DD:MM:SS'")
    d, m, sec = parts
    return sign * (d + m / 60.0 + sec / 3600.0)

def solved_filename(original_path):
    return original_path.replace(".fits", "_wcs.fit").replace(".fit", "_wcs.fit")

def try_solve(ast, infile, ra_deg, dec_deg, pixel_scale):
    """Try to solve with hints; returns wcs_header or None."""
    scale_lower = float(pixel_scale) * SCALE_MARGIN_LOW
    scale_upper = float(pixel_scale) * SCALE_MARGIN_HIGH
    # Use solve_from_image with scale & center hints
    return ast.solve_from_image(
        infile,
        scale_units='arcsecperpix',
        scale_lower=scale_lower,
        scale_upper=scale_upper,
        center_ra=ra_deg,
        center_dec=dec_deg,
        radius=RADIUS_DEG
    )

def crop_center_and_write(infile, outtemp, frac=0.5):
    """Crop the central frac x frac region and write to outtemp."""
    with fits.open(infile) as hdul:
        data = hdul[0].data
        if data is None:
            raise ValueError("No image data in FITS")
        ny, nx = data.shape
        cx0 = int((1 - frac) / 2 * ny)
        cx1 = int((1 + frac) / 2 * ny)
        cy0 = int((1 - frac) / 2 * nx)
        cy1 = int((1 + frac) / 2 * nx)
        sub = data[cx0:cx1, cy0:cy1]
        h = hdul[0].header.copy()
        # basic header write for the crop (no WCS yet)
        fits.writeto(outtemp, sub, header=h, overwrite=True)

# === Parse RA/DEC ===
try:
    RA_DEG = hms_to_deg(RA_STR)
    DEC_DEG = dms_to_deg(DEC_STR)
except Exception as e:
    raise SystemExit(f"Error parsing RA/DEC strings: {e}")

# === Prepare list of fits files ===
fits_files = []
for ext in extensions:
    fits_files.extend(sorted(glob(os.path.join(fits_dir, ext))))
fits_files = sorted(set(fits_files))
if not fits_files:
    raise SystemExit(f"No FITS files found in {fits_dir} (searched {extensions}).")

print(f"Found {len(fits_files)} FITS files. Using RA={RA_DEG:.6f} deg DEC={DEC_DEG:.6f} deg as hints.")
print(f"Pixel scale hint: {PIXEL_SCALE} arcsec/pixel (search range: {PIXEL_SCALE*SCALE_MARGIN_LOW:.3f} - {PIXEL_SCALE*SCALE_MARGIN_HIGH:.3f})")

# === AstrometryNet client ===
ast = AstrometryNet()
ast.api_key = API_KEY

solved_files = []

for f in fits_files:
    out_wcs = solved_filename(f)
    if os.path.exists(out_wcs):
        print(f"Skipping (already solved): {os.path.basename(f)}")
        solved_files.append(out_wcs)
        continue

    print(f"\nSolving WCS for: {os.path.basename(f)}")
    success = False
    last_exception = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            wcs_header = try_solve(ast, f, RA_DEG, DEC_DEG, PIXEL_SCALE)
            if wcs_header:
                # write header into new file
                with fits.open(f) as hdul:
                    hdul[0].header.update(wcs_header)
                    hdul.writeto(out_wcs, overwrite=True)
                print(f"✅ Solved and saved: {os.path.basename(out_wcs)}")
                solved_files.append(out_wcs)
                success = True
                break
            else:
                print(f"Attempt {attempt}: no wcs returned (server returned None).")
        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
            last_exception = e

        # small wait before retry
        if attempt < MAX_RETRIES:
            print(f"Retrying in {RETRY_WAIT} s...")
            sleep(RETRY_WAIT)

    # If still not solved, try a cropped fallback
    if not success:
        try:
            print("Trying cropped-center fallback (crop 50%) ...")
            tmpcrop = os.path.join(fits_dir, "._ast_crop_tmp.fit")
            crop_center_and_write(f, tmpcrop, frac=0.5)
            # try to solve the crop
            try:
                wcs_header = try_solve(ast, tmpcrop, RA_DEG, DEC_DEG, PIXEL_SCALE)
                if wcs_header:
                    # if crop solved, write header into original full image (safe to apply)
                    with fits.open(f) as hdul:
                        hdul[0].header.update(wcs_header)
                        hdul.writeto(out_wcs, overwrite=True)
                    print(f"✅ Cropped solve succeeded; saved WCS to {os.path.basename(out_wcs)}")
                    solved_files.append(out_wcs)
                    success = True
                else:
                    print("Cropped solve returned None.")
            except Exception as e:
                print(f"Cropped solve failed: {e}")
        except Exception as e:
            print(f"Could not create crop: {e}")

    if not success:
        print(f"❌ Failed to solve {os.path.basename(f)}. Last error: {last_exception}")

# === If nothing solved, stop ===
if not solved_files:
    raise SystemExit("No images were solved successfully. Try adjusting PIXEL_SCALE, RADIUS_DEG or provide more accurate RA/DEC.")

# === Reproject solved images to reference grid ===
ref_hdu = fits.open(solved_files[0])[0]
ref_wcs = WCS(ref_hdu.header)
ref_data = ref_hdu.data

for s in solved_files:
    print(f"Reprojecting {os.path.basename(s)} -> grid of {os.path.basename(solved_files[0])}")
    hdu = fits.open(s)[0]
    data, footprint = reproject_interp(hdu, ref_wcs, shape_out=ref_data.shape)
    out = os.path.join(output_dir, os.path.basename(s).replace("_wcs.fit", ".fit"))
    fits.writeto(out, data, ref_hdu.header, overwrite=True)
    print(f"Aligned saved: {out}")

print("\nAll done. Aligned files are in:", output_dir)
