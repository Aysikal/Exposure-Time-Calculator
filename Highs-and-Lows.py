import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import mad_std
from astropy.visualization import ZScaleInterval
from tabulate import tabulate
import random
import warnings

warnings.filterwarnings("ignore")

# --- File paths (edit these) ---
high_path = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Highs and Lows\u\area92_u_low2.8_high13.5_2025_10_08_2x2_exp00.02.00.000_000001_High_1.fit"
low_path  = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Highs and Lows\u\area92_u_low2.8_high13.5_2025_10_08_2x2_exp00.02.00.000_000001_Low_1.fit"

# --- Parameters you can tweak ---
box_size = 20          # half-size of square cutout (total = 2*box_size)
num_bg_regions = 10    # number of background patches to sample
bg_min_dist = 2 * box_size
max_attempts = 2000

# --- Load FITS data and gains ---
with fits.open(high_path) as hdul_high:
    high_data = hdul_high[0].data.astype(np.float32)
    gain_high = hdul_high[0].header.get('GAIN', np.nan)

with fits.open(low_path) as hdul_low:
    low_data = hdul_low[0].data.astype(np.float32)
    gain_low = hdul_low[0].header.get('GAIN', np.nan)

gain_ratio = gain_high / gain_low if gain_low not in (0, None, np.nan) else np.nan

# --- Interactive selection on the high image (ZScale) ---
selected_coords = []

def onclick(event):
    if event.inaxes and event.button == 1:  # left click
        x, y = int(event.xdata), int(event.ydata)
        selected_coords.append((x, y))
        ax.plot(x, y, 'ro', markersize=6)
        fig.canvas.draw()
        print(f"Selected: x={x}, y={y}")

print("Click on stars in the high image to select them. Close the window when finished.")
zscale = ZScaleInterval()
vmin, vmax = zscale.get_limits(high_data)

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(high_data, cmap='gray_r', origin='lower', vmin=vmin, vmax=vmax)
ax.set_title("Click to select stars (left-click). Close window when done.")
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
fig.canvas.mpl_disconnect(cid)

if len(selected_coords) == 0:
    raise SystemExit("No stars selected. Rerun and click on at least one star.")


print("\n📐 Gain Comparison:")
print(f"High Image GAIN = {gain_high}")
print(f"Low Image GAIN  = {gain_low}")
print(f"GAIN Ratio (High / Low) = {gain_ratio:.3f}")


print("\nFinal selected coordinates:")
for i, (x, y) in enumerate(selected_coords, 1):
    print(f"Star {i}: x={x}, y={y}")

# --- Star intensity ratios computation ---
ratios = []
for i, (x, y) in enumerate(selected_coords, 1):
    # bounds check
    y0, y1 = y - box_size, y + box_size
    x0, x1 = x - box_size, x + box_size
    if y0 < 0 or x0 < 0 or y1 > high_data.shape[0] or x1 > high_data.shape[1]:
        print(f"⚠️ Star {i}: cutout near edge — skipped (x={x}, y={y})")
        continue

    high_cutout = high_data[y0:y1, x0:x1]
    low_cutout  = low_data[y0:y1, x0:x1]

    if high_cutout.shape != (2*box_size, 2*box_size) or low_cutout.shape != (2*box_size, 2*box_size):
        print(f"⚠️ Star {i}: unexpected cutout shape — skipped")
        continue

    I_high = np.sum(high_cutout)
    I_low  = np.sum(low_cutout)
    ratio  = I_high / I_low if I_low != 0 else np.nan
    ratios.append((i, x, y, I_high, I_low, ratio))

print("\nIntensity Ratios (High / Low) for selected stars:")
for i, x, y, I_high, I_low, ratio in ratios:
    print(f"Star {i}: x={x}, y={y} | I_high={I_high:.2f} | I_low={I_low:.2f} | Ratio={ratio:.3f}")

# --- Background sampling (robust, star-free) ---
image_shape = high_data.shape
margin = box_size + 5
star_coords_set = set(selected_coords)

def is_far_from_stars(x, y, min_dist=bg_min_dist):
    for sx, sy in star_coords_set:
        if np.hypot(x - sx, y - sy) < min_dist:
            return False
    return True

bg_ratios = []
bg_locations = []
attempts = 0
rejected_bright = 0
rejected_structure = 0

while len(bg_ratios) < num_bg_regions and attempts < max_attempts:
    attempts += 1
    x = random.randint(margin, image_shape[1] - margin - 1)
    y = random.randint(margin, image_shape[0] - margin - 1)

    if not is_far_from_stars(x, y):
        continue

    y0, y1 = y - box_size, y + box_size
    x0, x1 = x - box_size, x + box_size
    if y0 < 0 or x0 < 0 or y1 > image_shape[0] or x1 > image_shape[1]:
        continue

    high_cutout = high_data[y0:y1, x0:x1]
    low_cutout  = low_data[y0:y1, x0:x1]

    if high_cutout.shape != (2*box_size, 2*box_size) or low_cutout.shape != (2*box_size, 2*box_size):
        continue

    # Bright spike filter (cosmic ray or faint star)
    if np.max(high_cutout) > np.median(high_cutout) + 5 * np.std(high_cutout):
        rejected_bright += 1
        continue

    # Structure filter (reject patches with too much texture)
    if np.std(high_cutout) > 2 * np.std(high_data):
        rejected_structure += 1
        continue

    I_high = np.sum(high_cutout)
    I_low  = np.sum(low_cutout)
    ratio  = I_high / I_low if I_low != 0 else np.nan

    bg_ratios.append(ratio)
    bg_locations.append((x, y))

if len(bg_ratios) == 0:
    raise SystemExit("No background regions found. Adjust parameters and try again.")

print(f"\nBackground Analysis: found {len(bg_ratios)} regions (attempts={attempts})")
print(f"Rejected: {rejected_bright} bright spikes, {rejected_structure} structured patches")
for i, ((x, y), ratio) in enumerate(zip(bg_locations, bg_ratios), 1):
    print(f"Region {i}: x={x}, y={y} | Ratio={ratio:.3f}")

# --- Statistics and table output ---
star_ratios = [r for (_, _, _, _, _, r) in ratios]
star_mean   = np.nanmean(star_ratios) if len(star_ratios) > 0 else np.nan
star_median = np.nanmedian(star_ratios) if len(star_ratios) > 0 else np.nan
star_std    = np.nanstd(star_ratios) if len(star_ratios) > 0 else np.nan

bg_mean     = np.nanmean(bg_ratios)
bg_median   = np.nanmedian(bg_ratios)
bg_std      = np.nanstd(bg_ratios)

summary_table = [
    ["Region Type", "Mean Ratio", "Median Ratio", "Std Dev"],
    ["Stars",       round(star_mean, 3), round(star_median, 3), round(star_std, 3)],
    ["Background",  round(bg_mean, 3),   round(bg_median, 3),   round(bg_std, 3)]
]

print("\nPhotometric Ratio Summary:\n")
print(tabulate(summary_table[1:], headers=summary_table[0], tablefmt="grid"))

# --- Optional: visualize a few background patches and star cutouts for confirmation ---
n_show = min(4, len(bg_locations))
plt.figure(figsize=(10, 4))
for i in range(n_show):
    x, y = bg_locations[i]
    cut = high_data[y-box_size:y+box_size, x-box_size:x+box_size]
    plt.subplot(2, n_show, i+1)
    plt.imshow(cut, cmap='gray_r', origin='lower', vmin=vmin, vmax=vmax)
    plt.title(f"BG {i+1}: x={x}, y={y}")
    plt.axis('off')

n_show_stars = min(4, len(ratios))
for i in range(n_show_stars):
    _, x, y, _, _, _ = ratios[i]
    cut = high_data[y-box_size:y+box_size, x-box_size:x+box_size]
    plt.subplot(2, n_show, n_show + i + 1)
    plt.imshow(cut, cmap='gray_r', origin='lower', vmin=vmin, vmax=vmax)
    plt.title(f"Star {i+1}: x={x}, y={y}")
    plt.axis('off')

plt.tight_layout()
plt.show()
