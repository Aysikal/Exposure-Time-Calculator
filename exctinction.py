# extinction_pipeline_fixed_aperture_netpos_filter_POSITIVE_K_UTCcheck.py
import os, math, csv, numpy as np, matplotlib.pyplot as plt
from datetime import datetime
import pytz
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.visualization import ZScaleInterval
from scipy.ndimage import gaussian_filter1d
from ancillary_functions import airmass_function
import matplotlib.patches as patches

# ---------------- CONFIG ----------------
ALIGNED_FOLDER = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\reduced\sept 30 area 95 g low\keep\aligned"
OUTPUT_TABLE_FOLDER = os.path.join(ALIGNED_FOLDER, "star_tables")
os.makedirs(OUTPUT_TABLE_FOLDER, exist_ok=True)

RA_HARD, DEC_HARD = "03:53:21", "-00:00:20"
PSF_ARCSEC = 0.7
PIXEL_SCALE = 0.047 * 1.8
BOX_FACTOR, REFINE_RADIUS_FACTOR = 10.0, 10.0
Z = ZScaleInterval()
K_AP, ANN_IN_FACT, ANN_OUT_FACT = 1.0, 3.0, 5.0
RADIAL_SMOOTH_SIGMA, TAIL_MEDIAN = 2.0, 15
SITE_TZ_NAME = "Asia/Tehran"
PLOT_SAVEFIG = False  # set True to save figures automatically

pixels_per_arcsec = PIXEL_SCALE
box_size_px = round((BOX_FACTOR * PSF_ARCSEC) / pixels_per_arcsec)
if box_size_px % 2 == 0:
    box_size_px += 1
PSF_PIX_REF = PSF_ARCSEC / PIXEL_SCALE
REFINE_BOX = int(round(REFINE_RADIUS_FACTOR * PSF_PIX_REF))
if REFINE_BOX % 2 == 0:
    REFINE_BOX += 1
FIXED_AP_RADIUS_PX = float(K_AP * PSF_PIX_REF)

print(f"Using fixed aperture {FIXED_AP_RADIUS_PX:.3f}px  box={box_size_px}, refine={REFINE_BOX}")

# ---------------- Helpers ----------------
def list_fits(folder):
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(('.fits', '.fit'))])

def load_fits(path):
    with fits.open(path, memmap=True) as hdul:
        return np.nan_to_num(hdul[0].data.astype(float)), hdul[0].header

def get_exptime_raw_from_header(hdr):
    for key in ("DURATION","EXPTIME","EXPOSURE","EXPTIM","ITIME","ONTIME","TELAPSE"):
        v = hdr.get(key)
        if v is not None:
            try: return float(v)
            except Exception:
                try: return float(str(v))
                except Exception: continue
    return None

def exptime_seconds_from_raw(raw_val):
    if raw_val is None: return np.nan
    try: return float(raw_val)*1e-5
    except Exception: return np.nan

def parse_dateobs_token(hdr):
    s = hdr.get('DATE-OBS')
    if s is None: return None
    s = str(s).split()[0]
    if "T" in s:
        d,t = s.split("T",1)
        t = ''.join(ch for ch in t if ch.isdigit() or ch in [":","."])
        return f"{d}T{t}"
    return s

def local_token_to_utc_components(token, tz_name=SITE_TZ_NAME):
    if token is None: return None, None, None, None
    try:
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f","%Y-%m-%dT%H:%M:%S"):
            try:
                dt_local = datetime.strptime(token, fmt)
                break
            except Exception: continue
        tz = pytz.timezone(tz_name)
        dt_local = tz.localize(dt_local)
        dt_utc = dt_local.astimezone(pytz.utc)
        return dt_utc.strftime("%Y-%m-%d"), dt_utc.hour, dt_utc.minute, dt_utc
    except Exception:
        return None, None, None, None

def centroid_in_array(arr):
    arr = np.nan_to_num(arr)
    if arr.size==0: return 0.,0.
    try:
        from photutils.centroids import centroid_com
        cy,cx = centroid_com(arr)
        return float(cx),float(cy)
    except Exception:
        total = arr.sum()
        if total<=0:
            i,j = np.unravel_index(np.nanargmax(arr), arr.shape)
            return float(j),float(i)
        yy,xx = np.indices(arr.shape)
        cx=(xx*arr).sum()/total; cy=(yy*arr).sum()/total
        return float(cx),float(cy)

def make_masks(shape, center, r_ap, r_in, r_out):
    yy,xx=np.indices(shape)
    rr=np.sqrt((xx-center[0])**2+(yy-center[1])**2)
    return rr<=r_ap, (rr>=r_in)&(rr<=r_out)

# ---------------- Interactive GUI selection ----------------
fits_files = list_fits(ALIGNED_FOLDER)
if not fits_files: raise SystemExit("No FITS files found.")
ref_data, ref_hdr = load_fits(os.path.join(ALIGNED_FOLDER,fits_files[0]))
clicked=[]
fig,ax=plt.subplots(figsize=(10,8))
vmin,vmax=Z.get_limits(ref_data)
ax.imshow(ref_data,cmap='gray',origin='lower',vmin=vmin,vmax=vmax)
ax.set_title(f"Click stars in {fits_files[0]}. Press Enter when done.")
def onclick(e):
    if e.inaxes:
        clicked.append((e.xdata,e.ydata))
        ax.plot(e.xdata,e.ydata,'o',color='lime',ms=8); fig.canvas.draw()
def onkey(e):
    if e.key in ('enter','return'):
        fig.canvas.mpl_disconnect(cid); fig.canvas.mpl_disconnect(kid); plt.close(fig)
cid=fig.canvas.mpl_connect('button_press_event',onclick)
kid=fig.canvas.mpl_connect('key_press_event',onkey)
plt.show()
if not clicked: raise SystemExit("No stars selected.")

refined=[]
for x,y in clicked:
    cut=Cutout2D(ref_data,(x,y),REFINE_BOX,mode='partial')
    cx,cy=centroid_in_array(cut.data)
    x0=x-cut.data.shape[1]/2; y0=y-cut.data.shape[0]/2
    refined.append((x0+cx,y0+cy))

# ---------------- Photometry + CSV + Fit ----------------
all_star_fits = []   # collect fits for combined overlay (must be before star loop)

for sid, (x_ref, y_ref) in enumerate(refined):

    print(f"\nProcessing star {sid+1} at ({x_ref:.2f},{y_ref:.2f})")

    records = []
    cutout_imgs = []
    cutout_centers = []
    cutout_names = []

    for idx, fname in enumerate(fits_files):
        data, hdr = load_fits(os.path.join(ALIGNED_FOLDER, fname))

        token = parse_dateobs_token(hdr)
        date_utc_str, hour_utc, minute_utc, dt_utc = local_token_to_utc_components(token)

        if date_utc_str is None:
            airmass_val = np.nan
            altitude_val = np.nan
        else:
            try:
                airmass_val = airmass_function(date_utc_str, hour_utc, minute_utc, RA_HARD, DEC_HARD, plot_night_curve=False)
                # approximate altitude from airmass via z = arccos(1/X) => alt = 90 - z (deg)
                altitude_val = 90.0 - math.degrees(math.acos(1.0/airmass_val)) if (airmass_val is not None and airmass_val>1.0) else 90.0
            except Exception:
                airmass_val = np.nan
                altitude_val = np.nan

        exptime_raw = get_exptime_raw_from_header(hdr)
        exptime_s = exptime_seconds_from_raw(exptime_raw)

        tiny_cut = Cutout2D(data, (x_ref, y_ref), REFINE_BOX, mode='partial')
        cx_local, cy_local = centroid_in_array(tiny_cut.data)
        dx, dy = cx_local - tiny_cut.data.shape[1]/2, cy_local - tiny_cut.data.shape[0]/2
        if math.hypot(dx, dy) > REFINE_BOX/2:
            cx_local, cy_local = tiny_cut.data.shape[1]/2, tiny_cut.data.shape[0]/2
        x_star, y_star = tiny_cut.to_original_position((cx_local, cy_local))

        disp_cut = Cutout2D(data, (x_star, y_star), box_size_px, mode='partial')
        disp_data = np.nan_to_num(disp_cut.data)
        cx_disp, cy_disp = disp_cut.to_cutout_position((x_star, y_star))

        cutout_imgs.append(disp_data)
        cutout_centers.append((cx_disp, cy_disp))
        cutout_names.append(os.path.basename(fname))

        ap_radius = FIXED_AP_RADIUS_PX
        ann_in, ann_out = ANN_IN_FACT*PSF_PIX_REF, ANN_OUT_FACT*PSF_PIX_REF
        ap_mask, ann_mask = make_masks(disp_data.shape, (cx_disp, cy_disp), ap_radius, ann_in, ann_out)
        ap_values, ann_values = disp_data[ap_mask], disp_data[ann_mask]
        sum_circle = float(np.nansum(ap_values))
        n_pix = int(np.count_nonzero(ap_mask))
        mean_ann = float(np.mean(ann_values)) if ann_values.size>0 else float(np.mean(disp_data))
        net = sum_circle - n_pix * mean_ann
        net_per_sec = net / exptime_s if (not np.isnan(exptime_s) and exptime_s>0) else np.nan
        # keep your existing magnitude zero offset (+15) if you want consistent numbers
        m_instr = -2.5*np.log10(net_per_sec) + 15 if (net_per_sec>0 and np.isfinite(net_per_sec)) else np.nan

        records.append({
            "dt_utc_obj": dt_utc,                        # for sorting internally
            "date_utc": date_utc_str or "",
            "time_utc": dt_utc.strftime("%H:%M:%S") if dt_utc else "",
            "file_name": os.path.basename(fname),
            "exptime_s": exptime_s,
            "sum_circle": sum_circle,
            "mean_annulus": mean_ann,
            "n_pix": n_pix,
            "net": net,
            "altitude_deg": altitude_val,
            "airmass": airmass_val,
            "instrumental_mag": m_instr
        })

    # ---- Sort and save CSV ----
    # sort by dt_utc_obj if present, otherwise fallback to time string
    records_sorted = sorted(records, key=lambda r: (r["dt_utc_obj"] if r["dt_utc_obj"] is not None else r["time_utc"]))
    csv_path = os.path.join(OUTPUT_TABLE_FOLDER, f"star_{sid+1:02d}_photometry.csv")
    # remove the internal dt_utc_obj when writing CSV (can't serialize datetime object)
    csv_fieldnames = [k for k in records_sorted[0].keys() if k != "dt_utc_obj"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        writer.writeheader()
        for row in records_sorted:
            row_out = {k: (v if k!="dt_utc_obj" else "") for k,v in row.items() if k != "dt_utc_obj"}
            writer.writerow(row_out)
    print(f"Saved CSV → {csv_path}")

    # ---- Plot all cutouts for this star ----
    n_frames = len(cutout_imgs)
    ncols = int(np.ceil(np.sqrt(n_frames)))
    nrows = int(np.ceil(n_frames / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.0 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for idx, data in enumerate(cutout_imgs):
        cx_disp, cy_disp = cutout_centers[idx]
        vmin, vmax = Z.get_limits(data)
        ax = axes[idx]
        ax.imshow(data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        circ_ap = patches.Circle((cx_disp, cy_disp), FIXED_AP_RADIUS_PX, edgecolor='lime', facecolor='none', lw=1.3)
        circ_in = patches.Circle((cx_disp, cy_disp), ANN_IN_FACT * PSF_PIX_REF, edgecolor='yellow', facecolor='none', lw=1.0, ls='--')
        circ_out = patches.Circle((cx_disp, cy_disp), ANN_OUT_FACT * PSF_PIX_REF, edgecolor='orange', facecolor='none', lw=1.0, ls='--')
        ax.add_patch(circ_ap); ax.add_patch(circ_in); ax.add_patch(circ_out)
        ax.set_title(f"{idx+1}: {cutout_names[idx]}", fontsize=7)
        ax.axis('off')
    # hide any extra axes
    for j in range(idx+1, len(axes)): axes[j].axis('off')
    plt.tight_layout()
    plt.suptitle(f"Star {sid+1}: all {n_frames} cutouts", y=1.02)
    if PLOT_SAVEFIG:
        out_path = os.path.join(OUTPUT_TABLE_FOLDER, f"star_{sid+1:02d}_cutouts.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved cutouts → {out_path}")
    else:
        plt.show(); plt.close(fig)

    # ---- Extinction fit for this star (sorted by airmass) ----
    airmass_arr = np.array([r["airmass"] for r in records_sorted], float)
    mag_arr = np.array([r["instrumental_mag"] for r in records_sorted], float)
    mask = np.isfinite(airmass_arr) & np.isfinite(mag_arr)
    if np.count_nonzero(mask) > 2:
        X = airmass_arr[mask].astype(float)
        Y = mag_arr[mask].astype(float)
        # sort by airmass ascending for stable plotting & fitting
        order = np.argsort(X)
        Xs = X[order]
        Ys = Y[order]
        coeffs = np.polyfit(Xs, Ys, 1)
        K, m0 = float(coeffs[0]), float(coeffs[1])
        Yfit_sorted = np.polyval(coeffs, Xs)

        # individual fit plot
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.scatter(Xs, Ys, s=30, label=f"Star {sid+1}")
        ax2.plot(Xs, Yfit_sorted, '--', lw=1.5, label=f"K={K:.3f}, m₀={m0:.3f}")
        ax2.set_xlabel("Airmass")
        ax2.set_ylabel("Instrumental magnitude")
        ax2.set_title(f"Star {sid+1}: Extinction Fit")
        ax2.legend()
        ax2.text(0.05, 0.9, r"$K\,\Delta X = \Delta m$", transform=ax2.transAxes,
                 fontsize=14, bbox=dict(facecolor="white", alpha=0.7, edgecolor="black"))
        if PLOT_SAVEFIG:
            out_fit = os.path.join(OUTPUT_TABLE_FOLDER, f"star_{sid+1:02d}_extinction_fit.png")
            plt.savefig(out_fit, dpi=150)
            plt.close(fig2)
            print(f"Saved fit → {out_fit}")
        else:
            plt.show(); plt.close(fig2)

        print(f"Extinction fit: Star {sid+1}  K={K:.4f}, m0={m0:.4f}")

        # store for combined overlay (use copies)
        all_star_fits.append({
            "sid": sid + 1,
            "X": Xs.copy(),
            "Y": Ys.copy(),
            "Xfit": Xs.copy(),
            "Yfit": Yfit_sorted.copy(),
            "K": K,
            "m0": m0
        })
    else:
        print(f"Star {sid+1}: not enough valid points for extinction fit (valid={np.count_nonzero(mask)})")

print("All stars processed.")

# ---- Combined overlay plot for all stars ----
if all_star_fits:
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    n = len(all_star_fits)
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, n)))

    for i, fit in enumerate(all_star_fits):
        color = colors[i % len(colors)]
        ax3.scatter(fit["X"], fit["Y"], s=25, color=color, alpha=0.85, label=f"Star {fit['sid']}")
        ax3.plot(fit["Xfit"], fit["Yfit"], '--', lw=1.5, color=color, label=f"Fit {fit['sid']}: K={fit['K']:.3f}")

    ax3.set_xlabel("Airmass")
    ax3.set_ylabel("Instrumental magnitude")
    ax3.set_title("All Stars: Extinction Fits Overlay")
    ax3.legend(fontsize=8, loc='upper left', ncol=2)
    ax3.text(0.5, 0.05, r"$K\,\Delta X = \Delta m$", transform=ax3.transAxes,
             fontsize=14, ha='center', va='center', bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"))
    ax3.grid(True)

    if PLOT_SAVEFIG:
        out_combined = os.path.join(OUTPUT_TABLE_FOLDER, "all_stars_extinction_overlay.png")
        plt.savefig(out_combined, dpi=150)
        plt.close(fig3)
        print(f"Saved combined overlay plot → {out_combined}")
    else:
        plt.show(); plt.close(fig3)
else:
    print("No valid extinction fits to plot in combined overlay.")
