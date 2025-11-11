import os, math
import numpy as np
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.visualization import ZScaleInterval
from datetime import datetime
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from astroplan import Observer
from astropy import units as u
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ---------------- CONFIG ----------------
FILTER_FOLDER = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\Rezaei_Hossein_Atanaz_Kosar_2025_11_04\light\Aysan\high\hot pixels removed\97b-8\aligned\g\reduced"
RA_HARD, DEC_HARD = "05:58:25.3", "+00:05:13.5"
PSF_ARCSEC = 1
PIXEL_SCALE = 0.101
REFINE_RADIUS_FACTOR = 20.0
Z = ZScaleInterval()
REFINE_BOX = int(round(REFINE_RADIUS_FACTOR * (PSF_ARCSEC/PIXEL_SCALE)))
if REFINE_BOX % 2 == 0:
    REFINE_BOX += 1

SITE_LAT, SITE_LON, SITE_ELEV = 35.674, 51.3188, 3600
SITE_LOCATION = EarthLocation(lat=SITE_LAT, lon=SITE_LON, height=SITE_ELEV*u.m)
SITE_OBSERVER = Observer(location=SITE_LOCATION, timezone="UTC", name="INO")

# Aperture scan settings
R_MIN = 1.0
R_MAX = 12.0
R_STEP = 0.5

# Fixed annulus geometry
ANNULUS_FACTOR = 2.4
ANNULUS_WIDTH  = 3.0

# Instrument parameters
GAIN = 1/45
READNOISE = 5.0

min_hwhm_pixels = 1.0
min_fwhm_pixels = 2.0
inner_radius_factor = 1.5
outer_radius_factor = 2.5

# ---------------- Airmass ----------------
def airmass_function(date_str, hour, minute, RA, DEC):
    dt_utc = datetime.strptime(f"{date_str} {hour:02d}:{minute:02d}", "%Y-%m-%d %H:%M")
    obs_time = Time(dt_utc)
    ra_h, ra_m, ra_s = map(float, RA.split(":"))
    ra_deg = ra_h*15 + ra_m*0.25 + ra_s*(0.25/60)
    d, m, s = map(float, DEC.split(":"))
    dec_sign = 1 if d >= 0 else -1
    dec_deg = dec_sign*(abs(d) + m/60 + s/3600)
    coord = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame="icrs")
    alt0 = coord.transform_to(AltAz(obstime=obs_time, location=SITE_LOCATION)).alt.degree
    z0 = np.radians(90 - alt0)
    X0 = 1.0 / (np.cos(z0) + 0.50572*(6.07995 + np.degrees(z0))**(-1.6364))
    return X0

# ---------------- Load ref frame ----------------
fits_files = sorted([f for f in os.listdir(FILTER_FOLDER) if f.lower().endswith((".fits", ".fit"))])
if not fits_files:
    raise SystemExit("No FITS files found.")

ref_file = os.path.join(FILTER_FOLDER, fits_files[0])
data, hdr = fits.getdata(ref_file), fits.getheader(ref_file)

# ---------------- Click stars ----------------
clicked = []
fig, ax = plt.subplots(figsize=(8, 8))
vmin, vmax = Z.get_limits(data)
ax.imshow(data, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
ax.set_title("Click stars, press Enter when done.")

def onclick(e):
    if e.inaxes:
        clicked.append((e.xdata, e.ydata))
        ax.plot(e.xdata, e.ydata, 'o', color='lime', ms=8)
        fig.canvas.draw()

def onkey(e):
    if e.key in ('enter', 'return'):
        fig.canvas.mpl_disconnect(cid)
        fig.canvas.mpl_disconnect(kid)
        plt.close(fig)

cid = fig.canvas.mpl_connect('button_press_event', onclick)
kid = fig.canvas.mpl_connect('key_press_event', onkey)
plt.show()

if not clicked:
    raise SystemExit("No stars selected.")

# ---------------- Refine star positions ----------------
refined = []
for (x, y) in clicked:
    cut = Cutout2D(data, (x, y), REFINE_BOX)
    cy, cx = np.unravel_index(np.argmax(cut.data), cut.data.shape)
    x0 = x - cut.data.shape[1]/2 + cx
    y0 = y - cut.data.shape[0]/2 + cy
    refined.append((x0, y0))

# ---------------- BEST-RADIUS FUNCTION ----------------
def get_radius(image, center_xy, HWHM, gain, readnoise,
               radius_step=0.5,
               inner_radius=inner_radius_factor, outer_radius=outer_radius_factor):
    cx, cy = center_xy
    yy, xx = np.indices(image.shape)
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    image = image.astype(np.float64)

    if not np.isfinite(HWHM) or HWHM <= 0:
        HWHM = 1.0
    HWHM = max(HWHM, min_hwhm_pixels)
    FWHM = max(2.0 * HWHM, min_fwhm_pixels)

    radius_min = max(1.0, 0.6 * FWHM)
    max_possible = max(1.0, min(image.shape)/2.0 - 1.0)
    radius_max = min(max_possible, max(radius_min + radius_step, 3.5 * FWHM, R_MAX))
    if radius_max <= radius_min:
        radius_max = radius_min + radius_step * 4.0

    best_radius = radius_min
    max_snr = -np.inf
    snrs, radii_list = [], []

    for radius in np.arange(radius_min, radius_max + radius_step/2.0, radius_step):
        star_mask = dist <= radius
        n_star_pix = np.count_nonzero(star_mask)
        if n_star_pix <= 0:
            snrs.append(0.0)
            radii_list.append(radius)
            continue

        sum_brightness = np.nansum(image[star_mask])
        ann_mask = (dist > inner_radius*radius) & (dist <= outer_radius*radius)
        ann_pixels = image[ann_mask]
        mean_bg = float(np.nanmedian(ann_pixels)) if ann_pixels.size > 0 else float(np.nanmedian(image))
        net_counts = sum_brightness - mean_bg*n_star_pix

        S_e = net_counts * gain
        sky_e = n_star_pix * mean_bg * gain
        var_e = max(S_e,0) + sky_e + n_star_pix * (readnoise**2)
        noise_e = np.sqrt(max(var_e, 1e-9))
        snr = S_e / noise_e if noise_e > 0 else 0.0

        snrs.append(float(snr))
        radii_list.append(float(radius))
        if snr > max_snr:
            max_snr = float(snr)
            best_radius = float(radius)

    if max_snr == -np.inf:
        max_snr = 0.0
        best_radius = radius_min

    return best_radius, max_snr, snrs, radii_list

# ---------------- Loop stars ----------------
for sid, (x_ref, y_ref) in enumerate(refined):
    mags, airmasses, snrs, exps = [], [], [], []
    cutout_images, best_radii, snr_lists_all, radii_lists_all = [], [], [], []

    for idx, fname in enumerate(fits_files):
        fpath = os.path.join(FILTER_FOLDER, fname)
        data_frame = fits.getdata(fpath)
        hdr_frame = fits.getheader(fpath)

        token = hdr_frame.get("DATE")
        dt_obj = datetime.fromisoformat(token)
        # Skip frames after 23:30 UTC (3 AM Tehran)
        if dt_obj.hour > 23 or (dt_obj.hour == 23 and dt_obj.minute >= 30):
            continue

        cut = Cutout2D(data_frame, (x_ref, y_ref), REFINE_BOX)
        cutout_images.append(cut.data)

        date_str = dt_obj.strftime("%Y-%m-%d")
        hour, minute = dt_obj.hour, dt_obj.minute
        X = airmass_function(date_str, hour, minute, RA_HARD, DEC_HARD)
        airmasses.append(X)

        exp = hdr_frame.get("EXPTIME", np.nan)
        exps.append(exp * 10**(-5))

        best_r, best_snr, snrs_list, radii_list = get_radius(
            data_frame, (x_ref, y_ref), HWHM=1.0, gain=GAIN, readnoise=READNOISE
        )
        best_radii.append(best_r)
        snrs.append(best_snr)
        snr_lists_all.append(snrs_list)
        radii_lists_all.append(radii_list)

        yy, xx = np.indices(data_frame.shape)
        dist = np.sqrt((xx-x_ref)**2 + (yy-y_ref)**2)
        flux = np.sum(data_frame[dist<=best_r])
        mag_inst = -2.5*np.log10(flux) if flux>0 else np.nan
        mags.append(mag_inst + 20)  # offset for display

    if len(cutout_images) == 0:
        print(f"Star {sid+1}: No valid frames before 3 AM Tehran, skipping.")
        continue

    # ----------- Show cutouts grid -----------
    N = len(cutout_images)
    ncols = int(np.ceil(np.sqrt(N)))
    nrows = int(np.ceil(N / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols,3*nrows))
    axes = np.array(axes).reshape(nrows,ncols)
    for i, ax in enumerate(axes.flat):
        if i < N:
            img = cutout_images[i]
            ax.imshow(img, origin='lower', cmap='gray')
            r = best_radii[i]
            cx, cy = img.shape[1]/2, img.shape[0]/2
            ax.add_patch(Circle((cx, cy), r, edgecolor='yellow', facecolor='none'))
            inner, outer = r*ANNULUS_FACTOR, r*ANNULUS_FACTOR + ANNULUS_WIDTH
            ax.add_patch(Circle((cx, cy), inner, edgecolor='cyan', facecolor='none', linestyle='--'))
            ax.add_patch(Circle((cx, cy), outer, edgecolor='cyan', facecolor='none', linestyle='--'))
            ax.set_title(f"{i+1}")
        else:
            ax.axis('off')
    plt.suptitle(f"Star {sid+1} cutouts")
    plt.tight_layout()
    plt.show()

    # ----------- SNR vs Radius grid -----------
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols,3*nrows))
    axes = np.array(axes).reshape(nrows,ncols)
    for i, ax in enumerate(axes.flat):
        if i < N:
            ax.plot(radii_lists_all[i], snr_lists_all[i], '-o')
            ax.axvline(best_radii[i], color='red', linestyle='--', label=f'Best r={best_radii[i]:.2f}')
            ax.set_xlabel("Radius (px)")
            ax.set_ylabel("SNR")
            ax.set_title(f"{i+1}")
            ax.grid(True)
            ax.legend()
        else:
            ax.axis('off')
    plt.suptitle(f"Star {sid+1}: SNR vs Radius")
    plt.tight_layout()
    plt.show()

    # ----------- Magnitude vs Airmass -----------
    X_arr, Y_arr = np.array(airmasses), np.array(mags)
    mask = np.isfinite(X_arr) & np.isfinite(Y_arr)
    if np.count_nonzero(mask) > 2:
        coeffs = np.polyfit(X_arr[mask], Y_arr[mask], 1)
        K, m0 = coeffs[0], coeffs[1]
        xfit = np.linspace(min(X_arr), max(X_arr), 100)
        yfit = K*xfit + m0
        print(f"Star {sid+1}: K={K:.4f}, m0={m0:.4f}")
        plt.figure(figsize=(6,4))
        plt.scatter(X_arr, Y_arr)
        plt.plot(xfit, yfit)
        plt.xlabel("Airmass X")
        plt.ylabel("Instrumental mag")
        plt.title(f"Star {sid+1}: Magnitude vs Airmass")
        plt.grid(True)
        plt.show()
    else:
        print(f"Star {sid+1}: Not enough points for fit.")
