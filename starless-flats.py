"""
make_starfree_master_flat.py

Requirements:
  pip install numpy astropy photutils scipy matplotlib

What it does:
  - Loads all .fits flats in folder
  - Detects bright/extended sources via photutils segmentation
  - Grows each source mask to cover the star (growing radius configurable)
  - Replaces masked pixels using the median of a surrounding annulus
  - Saves cleaned flats (optional) and produces a normalized master flat
"""

import os
import glob
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats, sigma_clip
from photutils import detect_threshold, detect_sources, deblend_sources
from photutils import SegmentationImage
from scipy.ndimage import binary_dilation
from scipy.ndimage import generate_binary_structure
from scipy.ndimage import distance_transform_edt
from scipy import ndimage

# ===== USER CONFIG =====
flat_path = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\data 2\Rezaei_Shakeri_Hashemi_etal_10_08_2025\flat_Hossein\r\high\hot pixels removed"
out_master_name = os.path.join(flat_path, "master_flat_starfree.fits")
save_cleaned_flats = False   # set True to save each cleaned flat (useful to inspect results)
cleaned_suffix = "_cleaned.fits"

# Detection parameters (tweak if necessary)
nsigma_detection = 4.0      # threshold = mean + nsigma_detection * std
npixels_min = 30            # minimum connected pixels to consider a source (large stars -> larger)
deblend = True              # attempt to deblend overlapping sources
deblend_nlevels = 32
deblend_contrast = 0.001

# Mask growth / replacement parameters
grow_radius = 6             # pixels to dilate mask by (for ~10px stars, 6 is usually safe)
annulus_inner = grow_radius + 1
annulus_width = 6           # width of annulus to sample background from

# Combination parameters
sigma_clip_sigma = 3.0
sigma_clip_iters = 5

# ========================

def circular_mask(shape, cy, cx, radius):
    Y, X = np.ogrid[:shape[0], :shape[1]]
    r2 = (X - cx)**2 + (Y - cy)**2
    return r2 <= radius**2

def annulus_indices(shape, cy, cx, r_in, r_out):
    Y, X = np.ogrid[:shape[0], :shape[1]]
    r2 = (X - cx)**2 + (Y - cy)**2
    mask = (r2 >= r_in**2) & (r2 <= r_out**2)
    return mask

def replace_mask_with_annulus_median(image, mask, label_img, seg_id):
    """
    For a given segment id in label_img, replace masked pixels in `image` that
    belong to that segment with the median of an annulus around its centroid.
    """
    seg_mask = (label_img == seg_id)
    if seg_mask.sum() == 0:
        return image  # nothing

    # compute centroid (approx)
    ys, xs = np.nonzero(seg_mask)
    cy = int(np.round(np.mean(ys)))
    cx = int(np.round(np.mean(xs)))

    # annulus
    r_in = annulus_inner
    r_out = annulus_inner + annulus_width
    ann_mask = annulus_indices(image.shape, cy, cx, r_in, r_out)

    # ensure we don't sample masked pixels
    sample = image[ann_mask & (~mask)]
    if sample.size < 10:
        # fallback: sample a larger box around centroid and exclude mask
        y0 = max(0, cy - (r_out + 10))
        y1 = min(image.shape[0], cy + (r_out + 10))
        x0 = max(0, cx - (r_out + 10))
        x1 = min(image.shape[1], cx + (r_out + 10))
        box = image[y0:y1, x0:x1]
        box_mask = mask[y0:y1, x0:x1]
        sample = box[~box_mask]
    if sample.size == 0:
        # last resort: global median of image
        fill_val = np.median(image[~mask])
    else:
        fill_val = np.median(sample)

    # fill segment pixels
    image[seg_mask] = fill_val
    mask[seg_mask] = False  # we've repaired them
    return image

def clean_flat(data, verbose=False):
    # estimate background and noise
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    thresh = median + nsigma_detection * std

    # detect sources (segmentation)
    segm = detect_sources(data, thresh, npixels=npixels_min)
    if segm is None:
        if verbose:
            print("No sources detected.")
        return data

    if deblend:
        segm = deblend_sources(data, segm, npixels=npixels_min,
                               nlevels=deblend_nlevels, contrast=deblend_contrast)

    label_img = segm.data.copy()  # integer labels

    # Create mask of all sources
    source_mask = label_img > 0

    # Grow mask to cover star wings
    structure = generate_binary_structure(2, 1)
    # Use distance transform to dilate by radius more precisely
    dt = distance_transform_edt(~source_mask)
    grown_mask = dt <= grow_radius
    # we'll operate on a copy
    clean = data.copy()
    mask = grown_mask.copy()

    # iterate over unique segments to replace each separately using annulus median
    seg_ids = np.unique(label_img)
    seg_ids = seg_ids[seg_ids > 0]
    # To speed up: sort by decreasing area (large first)
    areas = {sid: np.sum(label_img == sid) for sid in seg_ids}
    seg_ids_sorted = sorted(seg_ids, key=lambda s: -areas[s])

    for sid in seg_ids_sorted:
        # create segment mask (original seg) then dilate that segment by grow_radius
        seg_mask_orig = (label_img == sid)
        # dilate segment mask by binary_dilation with circular element
        # create circular footprint
        footprint = create_circular_footprint(grow_radius)
        seg_mask_grown = binary_dilation(seg_mask_orig, structure=footprint)
        # ensure we only affect pixels that are inside the global grown_mask
        seg_mask_grown = seg_mask_grown & mask

        if np.sum(seg_mask_grown) == 0:
            continue

        # replace pixels belonging to this segment
        clean = replace_mask_with_annulus_median(clean, mask, label_img, sid)

    return clean

def create_circular_footprint(radius):
    # returns boolean 2D structuring element of given integer radius
    r = int(np.ceil(radius))
    L = 2*r + 1
    Y, X = np.ogrid[-r:r+1, -r:r+1]
    footprint = (X**2 + Y**2) <= (radius**2)
    return footprint

def main():
    files = sorted(glob.glob(os.path.join(flat_path, "*.fit")))
    if len(files) == 0:
        print("No FITS files found in:", flat_path)
        return

    print(f"Found {len(files)} files. Cleaning each...")

    cleaned_stack = []
    for i, fpath in enumerate(files, 1):
        print(f"[{i}/{len(files)}] Loading {os.path.basename(fpath)}")
        with fits.open(fpath) as hdul:
            data = hdul[0].data.astype(float)
            header = hdul[0].header

        clean = clean_flat(data, verbose=True)

        if save_cleaned_flats:
            outname = os.path.splitext(fpath)[0] + cleaned_suffix
            fits.writeto(outname, clean, header=header, overwrite=True)
            print("  saved cleaned flat to", outname)

        cleaned_stack.append(clean)

    # combine with sigma clipping + median
    stack = np.array(cleaned_stack)
    clipped = sigma_clip(stack, sigma=sigma_clip_sigma, axis=0, maxiters=sigma_clip_iters)
    master = np.nanmedian(clipped, axis=0)

    # normalize
    med = np.median(master)
    if med == 0:
        print("Warning: median of master is zero; skipping normalization.")
    else:
        master = master / med

    fits.writeto(out_master_name, master, overwrite=True)
    print("Star-free master flat saved to:", out_master_name)

if __name__ == "__main__":
    main()
