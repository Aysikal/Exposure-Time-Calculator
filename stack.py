import os
from astroquery.astrometry_net import AstrometryNet
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
from glob import glob

# === CONFIG ===
fits_dir = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\reduced\Oct 7 area 95 green high\keep"
output_dir = os.path.join(fits_dir, "aligned")
os.makedirs(output_dir, exist_ok=True)

# === ASTROMETRY.NET SETUP ===
ast = AstrometryNet()
ast.api_key = "pakymwxyricoavhw"

# --- YOUR HINTS ---
PIXEL_SCALE = 0.047 * 1.8        # arcseconds per pixel (adjust to your setup)
RA_HINT = 53.123         # degrees (approximate telescope RA center)
DEC_HINT = +22.456       # degrees (approximate telescope DEC center)
RADIUS_DEG = 2           # search radius in degrees

fits_files = sorted(glob(os.path.join(fits_dir, "*.fit")))
solved_files = []

# === STEP 1: Solve WCS for each FIT ===
for f in fits_files:
    print(f"🔭 Solving WCS online for {f} ...")
    try:
        wcs_header = ast.solve_from_image(
            f,
            scale_units='arcsecperpix',
            scale_lower=PIXEL_SCALE * 0.8,
            scale_upper=PIXEL_SCALE * 1.2,
            center_ra=RA_HINT,
            center_dec=DEC_HINT,
            radius=RADIUS_DEG
        )
        if wcs_header:
            with fits.open(f) as hdul:
                hdul[0].header.update(wcs_header)
                out = f.replace(".fit", "_wcs.fit")
                hdul.writeto(out, overwrite=True)
                solved_files.append(out)
                print(f"✅ WCS solved and saved to {out}")
        else:
            print(f"❌ Could not solve {f}")
    except Exception as e:
        print(f"⚠️ Error solving {f}: {e}")

# === STEP 2: Reference and alignment ===
if not solved_files:
    raise RuntimeError("No solved files found!")

ref_hdu = fits.open(solved_files[0])[0]
ref_wcs = WCS(ref_hdu.header)
ref_data = ref_hdu.data

# === STEP 3: Reproject others ===
for f in solved_files:
    hdu = fits.open(f)[0]
    data, _ = reproject_interp(hdu, ref_wcs, shape_out=ref_data.shape)
    out = os.path.join(output_dir, os.path.basename(f))
    fits.writeto(out, data, ref_hdu.header, overwrite=True)
    print(f"🪄 Aligned: {out}")

print("🎯 All images aligned to reference grid!")
