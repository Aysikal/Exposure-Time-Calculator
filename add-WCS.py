from astropy.io import fits
import glob
import os

reference = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\Rezaei_Hossein_Atanaz_Kosar_2025_11_04\light\Aysan\high\hot pixels removed\97b-8\aligned\97b-8-wcs.fits"
targets_folder = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\Rezaei_Hossein_Atanaz_Kosar_2025_11_04\light\Aysan\high\hot pixels removed\97b-8\aligned\clear"

targets = glob.glob(os.path.join(targets_folder, "*.fit")) \
        + glob.glob(os.path.join(targets_folder, "*.fits"))

ref = fits.open(reference)
ref_header = ref[0].header

wcs_keywords = [
    k for k in ref_header
    if k.startswith(("CD", "CR", "CT", "PV", "CDELT",
                     "CUNIT", "CTYPE", "CRPIX", "LONG",
                     "LAT", "EQUINOX", "RADECSYS", "WCS"))
]

for t in targets:
    hdu = fits.open(t, mode="update")
    hdr = hdu[0].header
    for key in wcs_keywords:
        if key in ref_header:
            hdr[key] = ref_header[key]
    hdu.flush()
    hdu.close()
