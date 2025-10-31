import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import os

# Load and clean data
df = pd.read_excel(r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\plots\PSF\PSF distribution.xlsx")
df.columns = df.columns.str.strip().str.lower()
df['psf'] = pd.to_numeric(df['psf'], errors='coerce')
df.dropna(subset=['psf'], inplace=True)

# Create output folder
output_dir = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\plots\PSF\analysis"
os.makedirs(output_dir, exist_ok=True)

# Group and compute stats
grouped = df.groupby(['filter', 'mode'])
stats = grouped['psf'].agg(['mean', 'median', 'std', 'min', 'max']).reset_index()
stats.to_csv(os.path.join(output_dir, "psf_summary_statistics.csv"), index=False)

# Boxplot using matplotlib
filters = df['filter'].unique()
modes = ['High', 'Low']
fig, ax = plt.subplots(figsize=(10, 6))
positions = []
data = []
labels = []

for i, f in enumerate(filters):
    for j, m in enumerate(modes):
        subset = df[(df['filter'] == f) & (df['mode'] == m)]['psf']
        pos = i * 3 + j + 1
        positions.append(pos)
        data.append(subset)
        labels.append(f"{f}-{m}")

bp = ax.boxplot(data, positions=positions, patch_artist=True)
ax.set_xticks(positions)
ax.set_xticklabels(labels)
ax.set_title("PSF Distribution by Filter and Gain Mode")
ax.set_ylabel("PSF")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "psf_boxplot.png"))
plt.close()

# Histograms using matplotlib
for f in filters:
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in modes:
        subset = df[(df['filter'] == f) & (df['mode'] == m)]['psf']
        ax.hist(subset, bins=10, alpha=0.6, label=f"{m}", edgecolor='black')
    ax.set_title(f"PSF Histogram for Filter {f}")
    ax.set_xlabel("PSF")
    ax.set_ylabel("Frequency")
    ax.legend(title="Gain Mode")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# T-tests and effect size
def cohens_d(x, y):
    return (np.mean(x) - np.mean(y)) / np.sqrt((np.std(x)**2 + np.std(y)**2) / 2)

ttest_results = []
for f in filters:
    high_psf = df[(df['filter'] == f) & (df['mode'] == 'High')]['psf']
    low_psf = df[(df['filter'] == f) & (df['mode'] == 'Low')]['psf']
    t_stat, p_val = ttest_ind(high_psf, low_psf, equal_var=False)
    d = cohens_d(high_psf, low_psf)
    ttest_results.append({'filter': f, 't_stat': t_stat, 'p_value': p_val, 'cohens_d': d})

ttest_df = pd.DataFrame(ttest_results)
ttest_df.to_csv(os.path.join(output_dir, "psf_ttest_results.csv"), index=False)

# Print summary tables
print("Descriptive Statistics by Filter and Gain Mode:")
print(stats)
print("\nT-Test Results Comparing High vs Low Gain Modes:")
print(ttest_df)
