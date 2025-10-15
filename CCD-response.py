import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# File paths
files = {
    "Oct 1": r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\spreadsheets\dark_HIGH_Oct_1_2x2.csv",
    "Sept 30": r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\spreadsheets\dark_HIGH_Sept_30_2x2.csv",
    "Oct 7-8": r"C:\Users\AYSAN\Desktop\project\INO\ETC\Outputs\spreadsheets\dark_HIGH_Oct 7 and 8.csv",
}

# Load dataframes
dfs = {label: pd.read_csv(path) for label, path in files.items()}

# Metrics to cycle through on y axis
metrics = ["Mean", "Median", "StdDev"]

# Basic plot styles per night
styles = {
    "Oct 1": {"color": "C0", "marker": "o"},
    "Sept 30": {"color": "C1", "marker": "s"},
    "Oct 7-8": {"color": "C2", "marker": "D"},
}

# Loop over metrics: for each metric create per-night plots and one combined overlay
for metric in metrics:
    # Per-night individual plots
    for label, df in dfs.items():
        plt.figure(figsize=(8, 6))
        plt.scatter(df["ExposureTime"], df[metric], color=styles[label]["color"],
                    marker=styles[label]["marker"], label=label, alpha=0.8)
        plt.xlabel("Exposure Time")
        plt.ylabel(metric)
        plt.title(f"{label}: {metric} vs Exposure Time")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.show()

    # Combined overlay plot for this metric
    plt.figure(figsize=(8, 6))
    for label, df in dfs.items():
        plt.scatter(df["ExposureTime"], df[metric], color=styles[label]["color"],
                    marker=styles[label]["marker"], label=label, alpha=0.8)
    plt.xlabel("Exposure Time")
    plt.ylabel(metric)
    plt.title(f"{metric} vs Exposure Time (All Nights)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()


def positive_xy(df, xcol, ycol):
    mask = (df[xcol] > 0) & (df[ycol] > 0)
    return df.loc[mask, xcol].values, df.loc[mask, ycol].values

# Loop over metrics: per-night plots then combined overlay, all in log-log scale
for metric in metrics:
    # Individual per-night log-log plots
    for label, df in dfs.items():
        if "ExposureTime" not in df.columns or metric not in df.columns:
            print(f"Skipping {label} for {metric}: missing column")
            continue
        x, y = positive_xy(df, "ExposureTime", metric)
        if len(x) == 0:
            print(f"No positive points for {label} {metric}; skipping plot")
            continue
        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, color=styles[label]["color"], marker=styles[label]["marker"], alpha=0.8, label=label)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Exposure Time (s) log scale")
        plt.ylabel(metric + " log scale")
        plt.title(f"{label}: {metric} vs Exposure Time (log-log)")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Combined overlay log-log plot for this metric
    plt.figure(figsize=(8, 6))
    plotted_any = False
    for label, df in dfs.items():
        if "ExposureTime" not in df.columns or metric not in df.columns:
            continue
        x, y = positive_xy(df, "ExposureTime", metric)
        if len(x) == 0:
            continue
        plt.scatter(x, y, color=styles[label]["color"], marker=styles[label]["marker"], alpha=0.8, label=label)
        plotted_any = True

    if not plotted_any:
        print(f"No valid positive points across nights for metric {metric}; skipping combined plot")
        continue

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Exposure Time (s) log scale")
    plt.ylabel(metric + " log scale")
    plt.title(f"{metric} vs Exposure Time (All Nights) log-log")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()

print("Finished log-log plotting for Mean, Median, and StdDev.")
