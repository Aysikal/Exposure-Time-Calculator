from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
from astropy.table import Table
import numpy as np

# --- User config ---
center_rastr = "23 03 15.99847742006"
center_decstr = "-00 12 21.0136986066"
fov_arcmin = 6  # 8' x 8' -> radius ~4'
search_radius = (fov_arcmin / 2) * u.arcmin

GAIA_CATALOG = "I/350/gaiaedr3"
APASS_CATALOG = "II/336/apass9"

# --- center coordinate ---
coord = SkyCoord(center_rastr, center_decstr, unit=(u.hourangle, u.deg))

# --- query Gaia ---
vizier = Vizier(columns=["*"], row_limit=-1)
gaia_result = vizier.query_region(coord, radius=search_radius, catalog=GAIA_CATALOG)
if len(gaia_result) == 0:
    raise SystemExit("No Gaia results returned.")
gaia = gaia_result[0]

# Correct Gaia column names
ra_col = 'RA_ICRS' if 'RA_ICRS' in gaia.colnames else 'RAJ2000'
dec_col = 'DE_ICRS' if 'DE_ICRS' in gaia.colnames else 'DEJ2000'

# Make sure numeric
ra_vals = np.array(gaia[ra_col], dtype=float)
dec_vals = np.array(gaia[dec_col], dtype=float)
mask_finite = np.isfinite(ra_vals) & np.isfinite(dec_vals)

gaia_table = gaia[mask_finite]

# --- SkyCoord with numeric values only ---
gaia_coords = SkyCoord(ra=np.array(gaia_table[ra_col], dtype=float)*u.deg,
                       dec=np.array(gaia_table[dec_col], dtype=float)*u.deg)

print(f"Number of Gaia stars in FOV: {len(gaia_coords)}")

# Optional: query APASS
apass_result = vizier.query_region(coord, radius=search_radius, catalog=APASS_CATALOG)
if len(apass_result) > 0:
    apass = apass_result[0]
    print(f"Number of APASS stars: {len(apass)}")
else:
    apass = None

# --- Additional APASS inspection, filtering, cross-match, plotting, and save ---
if apass is None:
    print("No APASS table available to inspect.")
else:
    # 1) Show available columns and a quick preview
    print("APASS columns:", apass.colnames)
    print("APASS preview (first 10 rows):")
    print(apass[:10])

    # 2) Normalize common column name variations (safe, non-destructive)
    ra_apass_col = 'RAJ2000' if 'RAJ2000' in apass.colnames else ('RA_ICRS' if 'RA_ICRS' in apass.colnames else None)
    dec_apass_col = 'DEJ2000' if 'DEJ2000' in apass.colnames else ('DE_ICRS' if 'DE_ICRS' in apass.colnames else None)

    if ra_apass_col is None or dec_apass_col is None:
        print("APASS RA/Dec columns not found; cannot build SkyCoord for APASS.")
    else:
        # 3) Filter for finite positions and at least one magnitude (Vmag preferred)
        mag_cols = [c for c in ['Vmag','Bmag','gmag','rmag','imag'] if c in apass.colnames]
        pos_mask = np.isfinite(np.array(apass[ra_apass_col], dtype=float)) & np.isfinite(np.array(apass[dec_apass_col], dtype=float))
        if 'Vmag' in mag_cols:
            phot_mask = np.isfinite(np.array(apass['Vmag'], dtype=float))
        else:
            phot_mask = np.ones(len(apass), dtype=bool)

        good_mask = pos_mask & phot_mask
        apass_clean = apass[good_mask]
        print(f"APASS rows total: {len(apass)}, after position+phot filtering: {len(apass_clean)}")

        # 4) Create SkyCoord for APASS cleaned table
        apass_coords = SkyCoord(ra=np.array(apass_clean[ra_apass_col], dtype=float)*u.deg,
                                dec=np.array(apass_clean[dec_apass_col], dtype=float)*u.deg)

        # 5) Optional quick histogram of V magnitudes if present
        try:
            import matplotlib.pyplot as plt
            if 'Vmag' in apass_clean.colnames and np.any(np.isfinite(apass_clean['Vmag'])):
                plt.figure(figsize=(6,4))
                plt.hist(np.array(apass_clean['Vmag'], dtype=float), bins=30, color='skyblue', edgecolor='k')
                plt.xlabel('V magnitude')
                plt.ylabel('Number of stars')
                plt.title('APASS V magnitude distribution')
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.show()
            else:
                print("No Vmag available for histogram.")
        except Exception as e:
            print("Matplotlib plotting skipped or failed:", e)

        # 6) Cross-match APASS cleaned to Gaia (1 arcsec default tolerance)
        try:
            from astropy.coordinates import match_coordinates_sky
            idx, sep2d, _ = match_coordinates_sky(apass_coords, gaia_coords)
            tol = 1.0 * u.arcsec
            matched_mask = sep2d < tol
            n_matched = np.sum(matched_mask)
            print(f"APASS -> Gaia matches within {tol.to(u.arcsec).value:.1f}\" : {n_matched} / {len(apass_clean)}")

            # Build a combined table of matched entries (APASS + Gaia) for downstream use
            if n_matched > 0:
                gaia_matches = gaia_table[idx[matched_mask]]
                apass_matches = apass_clean[matched_mask]

                # Prefix columns to avoid name collisions
                from astropy.table import hstack, Column
                # add small separation column
                apass_matches = apass_matches.copy()
                apass_matches.add_column(Column(sep2d[matched_mask].arcsec, name='sep_arcsec'))

                # trim Gaia to the same rows used earlier (gaia_table used to create gaia_coords)
                matched_combined = hstack([apass_matches, gaia_matches], join_type='exact')

                print("Combined matched table preview:")
                print(matched_combined[:10])
                # Save matched table
                matched_combined.write("apass_gaia_matched.csv", format="csv", overwrite=True)
                print("Saved matched table to apass_gaia_matched.csv")
            else:
                print("No matches found; no combined table created.")
        except Exception as e:
            print("Cross-match failed:", e)

        # 7) Save cleaned APASS table for your records
        try:
            apass_clean.write("apass_clean.csv", format="csv", overwrite=True)
            print("Saved cleaned APASS table to apass_clean.csv")
        except Exception as e:
            print("Failed to save APASS cleaned table:", e)
# --- Visual check and photometry confirmation for a specific APASS star ---
from astropy import units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astroquery.vizier import Vizier
from astroquery.skyview import SkyView
import matplotlib.pyplot as plt
import numpy as np

# Replace these with the APASS row values you want to inspect (deg)
apass_ra = 345.814836
apass_dec = -0.205579

# build coord
target_coord = SkyCoord(ra=apass_ra * u.deg, dec=apass_dec * u.deg)

# 1) Fetch image cutouts (DSS2 color and 2MASS J) for visual confirmation
#    SkyView returns FITS image(s); we request small size ~4 arcmin
size = 4 * u.arcmin
surveys = ['DSS2 color', '2MASS-J']
images = SkyView.get_images(position=target_coord, survey=surveys, coordinates='J2000', width=size, height=size)

# display images with the target marker
fig, axes = plt.subplots(1, len(images), figsize=(6 * len(images), 6))
if len(images) == 1:
    axes = [axes]
for ax, hdu, survey in zip(axes, images, surveys):
    data = hdu[0].data
    header = hdu[0].header
    # simple display: flip if needed and use imshow; autoscale with percentile
    vmin, vmax = np.percentile(data[np.isfinite(data)], [2, 98])
    ax.imshow(data, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
    # mark the APASS position: convert world to pixel using WCS if available
    try:
        from astropy.wcs import WCS
        wcs = WCS(header)
        xpix, ypix = wcs.world_to_pixel(target_coord)
        ax.scatter(xpix, ypix, s=120, edgecolor='red', facecolor='none', lw=1.6)
        ax.set_title(f"{survey} (marker = APASS pos)")
    except Exception:
        ax.set_title(f"{survey} (no WCS)")
        ax.scatter(data.shape[1]/2, data.shape[0]/2, s=120, edgecolor='red', facecolor='none', lw=1.6)
    ax.set_axis_off()
plt.tight_layout()
plt.show()

# 2) Query nearby catalogs for photometry to confirm magnitude
viz = Vizier(columns=['*'], row_limit=50)

# Query APASS around the exact coordinate with small radius to pick the same source
r = 5 * u.arcsec
apass_q = viz.query_region(target_coord, radius=r, catalog='II/336/apass9')
if len(apass_q) > 0:
    apass_near = apass_q[0]
    print("APASS entries near target (within 5\"):")
    for row in apass_near:
        print(row)
else:
    print("No APASS entry returned at that exact position.")

# Query Gaia EDR3 near the same position
gaia_q = viz.query_region(target_coord, radius=r, catalog='I/350/gaiaedr3')
if len(gaia_q) > 0:
    gaia_near = gaia_q[0]
    print("\nGaia entries near target (within 5\"):")
    for row in gaia_near:
        # print a concise set of columns if present
        cols = {}
        for c in ['RA_ICRS','DE_ICRS','Gmag','BPmag','RPmag','pmRA','pmDE','Plx']:
            if c in gaia_near.colnames:
                cols[c] = row[c]
        print(cols)
else:
    print("No Gaia entry returned at that exact position.")

# Try Pan-STARRS (PS1) via Vizier (if available)
try:
    ps1_q = viz.query_region(target_coord, radius=r, catalog='II/349/ps1')
    if len(ps1_q) > 0:
        ps1_near = ps1_q[0]
        print("\nPan-STARRS (PS1) entries near target (within 5\"):")
        for row in ps1_near:
            # PS1 column names vary; common ones: gmag, rmag, imag
            cols = {}
            for c in ['gmag','rmag','imag','gMeanPSFMag','rMeanPSFMag','iMeanPSFMag']:
                if c in ps1_near.colnames:
                    cols[c] = row[c]
            print(cols)
    else:
        print("No Pan-STARRS entries returned at that exact position.")
except Exception as e:
    print("Pan-STARRS query skipped/failed:", e)

# 3) If multiple catalog entries are returned, cross-match by nearest coordinate and show matched mags
# Build small combined check list
collected = []

# from APASS
if len(apass_q) > 0:
    for row in apass_q[0]:
        collected.append(("APASS", float(row['RAJ2000']), float(row['DEJ2000']),
                          {k: row[k] for k in ['Vmag','Bmag'] if k in row.colnames}))

# from Gaia
if len(gaia_q) > 0:
    for row in gaia_q[0]:
        got = {}
        for k in ['Gmag','BPmag','RPmag']:
            if k in row.colnames:
                got[k] = row[k]
        collected.append(("Gaia", float(row['RA_ICRS']) if 'RA_ICRS' in row.colnames else float(row['RAJ2000']),
                          float(row['DE_ICRS']) if 'DE_ICRS' in row.colnames else float(row['DEJ2000']),
                          got))

# from PS1
if 'ps1_near' in locals() and len(ps1_q) > 0:
    for row in ps1_q[0]:
        got = {}
        for k in ['gmag','rmag','imag','gMeanPSFMag','rMeanPSFMag','iMeanPSFMag']:
            if k in row.colnames:
                got[k] = row[k]
        collected.append(("PS1", float(row['RAJ2000']) if 'RAJ2000' in row.colnames else float(row['RAICRS']),
                          float(row['DEJ2000']) if 'DEJ2000' in row.colnames else float(row['DEICRS']),
                          got))

# Print collected photometry with separation from your APASS position
print("\nCollected photometry near APASS pos:")
for cat, ra_c, dec_c, mags in collected:
    c = SkyCoord(ra=ra_c*u.deg, dec=dec_c*u.deg)
    sep = c.separation(target_coord).arcsec
    print(f"{cat:6s}  sep={sep:5.2f}\"  mags={mags}")

# Conclusion: you can compare APASS Vmag (from the APASS row) to Gaia G and PS1 mags printed above.
# If you want a numeric magnitude comparison or a small report, tell me which catalog mag you want compared to APASS V.
