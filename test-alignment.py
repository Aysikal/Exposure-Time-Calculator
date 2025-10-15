"""
Robust alignment + stacking script (RANSAC-style)
- Detect stars with DAOStarFinder
- Filter stars by local brightness (simple quality metric)
- Pair stars by bidirectional nearest neighbors
- Use a RANSAC loop to find the best similarity transform (robust to bad matches)
- Apply a center-corrected transform and warp with edge padding
- Save only valid aligned FITS and produce a stacked.fits

Requirements:
  astropy, photutils, scipy, scikit-image, numpy
"""

import os
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from scipy.spatial import cKDTree
from skimage.transform import estimate_transform, warp, SimilarityTransform

# ---------------- CONFIG (tweak these for your data) ----------------
FOLDER = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\Sep30\Rezaei_30_sep_2025\target3\g\low\1 min"
OUT_FOLDER = os.path.join(FOLDER, "aligned")
STACK_PATH = os.path.join(FOLDER, "stacked.fits")
os.makedirs(OUT_FOLDER, exist_ok=True)

# Detection / matching parameters
FWHM = 8                   # expected star FWHM in px (tweak)
SIGMA_THRESH = 7         # detection threshold (sigma)
BRIGHTNESS_PERCENTILE = 30 # keep stars brighter than this percentile (quality filter)
MAX_MATCH_DIST = 400       # max pixel distance to consider a match
MIN_MATCHES = 6            # minimum initial matched pairs required
TOP_N_MATCHES = 80         # keep up to this many candidate matches for RANSAC

# RANSAC / transform parameters
RANSAC_ITERS = 400         # how many random trials
RANSAC_INLIER_THRESH = 6.0 # px threshold to count an inlier (after transform)
RANSAC_MIN_INLIERS = 6     # require at least this many inliers for successful model
MAX_RMS_ERROR = 60.0       # final RMS acceptance (px)

# Warping
WARP_ORDER = 3             # bicubic
WARP_MODE = 'edge'         # fill edges with nearest values to avoid NaNs

# ---------------- Helpers ----------------
def load_fits(path):
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data.astype(float)
        hdr = hdul[0].header
    return data, hdr

def detect_stars(image, fwhm, sigma_thresh):
    """Detect centroids using DAOStarFinder. Returns Nx2 array of (x,y)."""
    mean, med, std = sigma_clipped_stats(image, sigma=3.0)
    finder = DAOStarFinder(threshold=sigma_thresh * std, fwhm=fwhm)
    sources = finder(image - med)
    if sources is None or len(sources) == 0:
        return np.empty((0,2))
    # photutils DAOStarFinder returns xcentroid, ycentroid
    pts = np.vstack([sources['xcentroid'], sources['ycentroid']]).T
    return pts, sources  # return table as well to access flux if available

def local_brightness(image, pts, aperture=3):
    """Simple local brightness: sum in (2*aperture+1)^2 box around centroid."""
    h, w = image.shape
    b = []
    for x, y in pts:
        ix = int(round(x))
        iy = int(round(y))
        x0 = max(0, ix - aperture)
        x1 = min(w, ix + aperture + 1)
        y0 = max(0, iy - aperture)
        y1 = min(h, iy + aperture + 1)
        sub = image[y0:y1, x0:x1]
        if sub.size == 0:
            b.append(0.0)
        else:
            # subtract local median to reduce background bias
            b.append(np.sum(sub - np.median(sub)))
    return np.array(b)

def bidirectional_match(ref_coords, tgt_coords, max_dist):
    """Bidirectional nearest neighbor matching. Returns ref_matched, tgt_matched Nx2 arrays."""
    if len(ref_coords) == 0 or len(tgt_coords) == 0:
        return np.empty((0,2)), np.empty((0,2))
    tree_ref = cKDTree(ref_coords)
    tree_tgt = cKDTree(tgt_coords)
    matches = []
    for i, pt_ref in enumerate(ref_coords):
        dist, idx_tgt = tree_tgt.query(pt_ref, distance_upper_bound=max_dist)
        if dist < max_dist and idx_tgt < len(tgt_coords):
            pt_tgt = tgt_coords[idx_tgt]
            dist_back, idx_back = tree_ref.query(pt_tgt, distance_upper_bound=max_dist)
            if idx_back == i:
                matches.append((pt_ref, pt_tgt))
    if not matches:
        return np.empty((0,2)), np.empty((0,2))
    ref_m, tgt_m = zip(*matches)
    return np.array(ref_m), np.array(tgt_m)

def compute_rms(a, b):
    """RMS between two Nx2 arrays of coordinates."""
    return np.sqrt(np.mean(np.sum((a - b)**2, axis=1)))

def center_transform(tform, ref_shape):
    """Return a transform that rotates/scales about the image center."""
    # ref_shape is (ny, nx)
    ny, nx = ref_shape
    cx, cy = nx/2.0, ny/2.0
    # shift_to_origin: translate center to origin (note SimilarityTransform uses (tx, ty) as x,y)
    shift_to_origin = SimilarityTransform(translation=[-cx, -cy])
    shift_back = SimilarityTransform(translation=[cx, cy])
    # compose: shift_back + tform + shift_to_origin (skimage comp order)
    return shift_back + tform + shift_to_origin

def apply_warp_centered(tgt_data, tform, ref_shape, order=WARP_ORDER, mode=WARP_MODE):
    """Apply center-corrected warp and return aligned array."""
    centered = center_transform(tform, ref_shape)
    aligned = warp(
        tgt_data,
        inverse_map=centered.inverse,
        output_shape=ref_shape,
        order=order,
        preserve_range=True,
        mode=mode,
        cval=0.0
    )
    return aligned

# RANSAC-like robust estimator
def ransac_similarity(ref_pts, tgt_pts, n_iter=RANSAC_ITERS, inlier_thresh=RANSAC_INLIER_THRESH, min_inliers=RANSAC_MIN_INLIERS):
    """
    Given matched coordinates (ref_pts, tgt_pts) in (x,y),
    run a simple RANSAC to find best similarity transform.
    Returns: best_tform (skimage transform) or None, inlier_mask (bool array)
    """
    N = len(ref_pts)
    if N < 2:
        return None, np.zeros(N, dtype=bool)

    best_inliers = np.zeros(N, dtype=bool)
    best_count = 0
    best_tform = None

    # convert to row,col for skimage estimate (y,x)
    ref_rc_all = ref_pts[:, ::-1]
    tgt_rc_all = tgt_pts[:, ::-1]

    rng = np.random.default_rng()

    # limit iterations if very few matches
    n_iter = max(50, min(n_iter, 2000))

    for it in range(n_iter):
        # sample minimal subset: similarity needs at least 2 points (but 3 better). We'll use 3 if available.
        k = 3 if N >= 3 else 2
        sample_idx = rng.choice(N, size=k, replace=False)
        ref_sample = ref_rc_all[sample_idx]
        tgt_sample = tgt_rc_all[sample_idx]

        try:
            t = estimate_transform('similarity', tgt_sample, ref_sample)
        except Exception:
            continue

        # transform all target points
        transformed = t(tgt_rc_all)  # returns (row, col) pairs
        # re-convert to (x,y) for distance measure consistent with ref_pts? we'll measure in rc coords (consistent)
        diffs = np.sqrt(np.sum((transformed - ref_rc_all)**2, axis=1))
        inliers = diffs < inlier_thresh
        count = np.count_nonzero(inliers)

        # require at least min_inliers and pick best by count, then by RMS
        if count >= min_inliers:
            # compute RMS on inliers
            rms_in = np.sqrt(np.mean(diffs[inliers]**2)) if count > 0 else np.inf
            # prefer larger counts, or equal counts but lower rms
            if count > best_count or (count == best_count and (best_tform is None or rms_in < best_rms)):
                best_count = count
                best_inliers = inliers
                best_tform = t
                best_rms = rms_in

    if best_tform is None:
        return None, np.zeros(N, dtype=bool)

    return best_tform, best_inliers

# ---------------- Main processing ----------------
def main():
    fits_files = sorted([f for f in os.listdir(FOLDER) if f.lower().endswith(('.fit', '.fits'))])
    if len(fits_files) < 2:
        raise SystemExit("Need at least two FITS files in the folder.")

    ref_index = len(fits_files) // 2
    ref_path = os.path.join(FOLDER, fits_files[ref_index])
    ref_data, ref_hdr = load_fits(ref_path)

    # Detect stars in reference and compute brightness-based quality
    ref_pts_all, ref_table = detect_stars(ref_data, FWHM, SIGMA_THRESH)
    if ref_pts_all.size == 0:
        raise SystemExit("No stars found in reference image.")

    # compute brightness metric and filter low-quality stars
    ref_brightness = local_brightness(ref_data, ref_pts_all, aperture=3)
    threshold = np.percentile(ref_brightness, BRIGHTNESS_PERCENTILE)
    good_ref_mask = ref_brightness > threshold
    ref_pts = ref_pts_all[good_ref_mask]
    print(f"Reference: {fits_files[ref_index]} — {len(ref_pts_all)} detected, {len(ref_pts)} kept after brightness filter")

    aligned_images = []
    summary = []  # list of tuples (fname, status, n_detected, n_matched, n_inliers, rms)

    for fname in fits_files:
        path = os.path.join(FOLDER, fname)
        tgt_data, tgt_hdr = load_fits(path)

        if fname == fits_files[ref_index]:
            aligned_images.append(tgt_data.copy())
            summary.append((fname, 'reference', len(ref_pts_all), 0, len(ref_pts), 0.0))
            print(f"{fname}: reference (kept)")
            continue

        # detect stars in target
        tgt_pts_all, tgt_table = detect_stars(tgt_data, FWHM, SIGMA_THRESH)
        if tgt_pts_all.size == 0:
            summary.append((fname, 'no_stars', 0, 0, 0, np.nan))
            print(f"{fname}: no stars detected — skipped")
            continue

        # brightness filter on target
        tgt_brightness = local_brightness(tgt_data, tgt_pts_all, aperture=3)
        thresh_tgt = np.percentile(tgt_brightness, BRIGHTNESS_PERCENTILE)
        tgt_good_mask = tgt_brightness > thresh_tgt
        tgt_pts = tgt_pts_all[tgt_good_mask]

        # bidirectional match between filtered star lists
        ref_matched, tgt_matched = bidirectional_match(ref_pts, tgt_pts, MAX_MATCH_DIST)
        n_matched = len(ref_matched)
        if n_matched < MIN_MATCHES:
            summary.append((fname, 'too_few_matches', len(tgt_pts_all), n_matched, 0, np.nan))
            print(f"{fname}: too few matched stars ({n_matched}) — skipped")
            continue

        # optionally limit to top candidate matches (by separation)
        sep = np.linalg.norm(ref_matched - tgt_matched, axis=1)
        if len(sep) > TOP_N_MATCHES:
            idx_keep = np.argsort(sep)[:TOP_N_MATCHES]
            ref_matched = ref_matched[idx_keep]
            tgt_matched = tgt_matched[idx_keep]

        # run RANSAC-style robust fit
        tform, inlier_mask = ransac_similarity(ref_matched, tgt_matched,
                                              n_iter=RANSAC_ITERS,
                                              inlier_thresh=RANSAC_INLIER_THRESH,
                                              min_inliers=RANSAC_MIN_INLIERS)
        if tform is None:
            summary.append((fname, 'ransac_failed', len(tgt_pts_all), n_matched, 0, np.nan))
            print(f"{fname}: RANSAC failed to find a good transform — skipped")
            continue

        n_inliers = int(np.count_nonzero(inlier_mask))
        # compute RMS on inliers (use rc coords)
        ref_rc = ref_matched[:, ::-1]
        tgt_rc = tgt_matched[:, ::-1]
        transformed = tform(tgt_rc)
        diffs = np.sqrt(np.sum((transformed - ref_rc)**2, axis=1))
        if n_inliers > 0:
            rms_all = np.sqrt(np.mean(diffs[inlier_mask]**2))
            median_in = np.median(diffs[inlier_mask])
        else:
            rms_all = np.inf
            median_in = np.inf

        # final accept/reject based on RMS and minimum inliers
        if n_inliers < RANSAC_MIN_INLIERS or rms_all > MAX_RMS_ERROR:
            summary.append((fname, 'rejected_quality', len(tgt_pts_all), n_matched, n_inliers, float(rms_all)))
            print(f"{fname}: rejected (inliers={n_inliers}, rms={rms_all:.2f}) — skipped")
            continue

        # Apply centered warp
        aligned = apply_warp_centered(tgt_data, tform, ref_data.shape, order=WARP_ORDER, mode=WARP_MODE)

        # sanity check: not all zero or NaN
        if np.all(np.isnan(aligned)) or np.nanmax(aligned) == 0:
            summary.append((fname, 'empty_after_warp', len(tgt_pts_all), n_matched, n_inliers, float(rms_all)))
            print(f"{fname}: produced empty image after warp — skipped")
            continue

        # save aligned FITS
        out_name = os.path.join(OUT_FOLDER, f"aligned_{fname}")
        fits.PrimaryHDU(aligned, header=tgt_hdr).writeto(out_name, overwrite=True)

        aligned_images.append(aligned)
        summary.append((fname, 'aligned', len(tgt_pts_all), n_matched, n_inliers, float(rms_all)))
        print(f"{fname}: aligned ✅ (matched={n_matched}, inliers={n_inliers}, rms={rms_all:.2f})")

    # stack
    if aligned_images:
        cube = np.stack(aligned_images, axis=0)
        stacked = np.nanmean(cube, axis=0)
        fits.PrimaryHDU(stacked, header=ref_hdr).writeto(STACK_PATH, overwrite=True)
        print(f"\nStacked image saved at: {STACK_PATH}")
    else:
        print("\nNo aligned images to stack.")

    # print summary table
    print("\nSummary:")
    print("{:<60s} {:<18s} {:>6s} {:>8s} {:>8s} {:>8s}".format("filename", "status", "ndet", "nmatch", "ninlrs", "rms"))
    for fn, status, nd, nm, ni, rms in summary:
        print(f"{fn:<60s} {status:<18s} {nd:6d} {nm:8d} {ni:8d} {rms:8.2f}")

if __name__ == "__main__":
    main()
