#!/usr/bin/env python3
"""
Inventory FITS files and write chronological CSV plus an English summary report.

Outputs:
- CSV at OUT_CSV with columns: file_path, object_name, filter, date, time_utc, time_local, exposure_seconds
- Summary printed to stdout and saved to OUT_SUMMARY (text)

Summary includes:
- total number of files
- overall capture time range (UTC) with 1-decimal seconds
- per-object counts and time ranges
- per-filter counts
- exposure statistics (min, mean, median, max) in seconds
"""
import os
import re
import csv
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict, Counter
import statistics

from astropy.io import fits
from dateutil import parser as dtparse
from dateutil import tz

# === CONFIG ===
ROOT_DIR = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\rezaei_saba_farideH_2025_10_22"
OUT_CSV = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\rezaei_saba_farideH_2025_10_22\rezaei_saba_farideH_2025_10_22_inventory.csv"
OUT_SUMMARY = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\rezaei_saba_farideH_2025_10_22\rezaei_saba_farideH_2025_10_22_summary.txt"
LOCAL_TZ = tz.gettz("Asia/Tehran")
# filename pattern: object_filter_..._YYYY_MM_DD...
FNAME_DATE_RE = re.compile(r"(?P<object>[^_]+)_(?P<filter>[A-Za-z0-9]+)_[^_]*_(?P<date>\d{4}_\d{2}_\d{2})")
EXPT_KEYS = ("EXPTIME", "EXPOSURE", "EXPO", "EXPOS")
# === end config ===

def parse_filename_info(filename: str):
    m = FNAME_DATE_RE.search(filename)
    if not m:
        return "", "", ""
    return m.group("object"), m.group("filter"), m.group("date").replace("_", "-")

def safe_open_fits(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            with fits.open(path, mode="readonly", ignore_missing_end=True, memmap=False) as hdul:
                return hdul[0].header, None
        except Exception as e1:
            try:
                with fits.open(path, mode="readonly", ignore_missing_end=True, memmap=True) as hdul:
                    return hdul[0].header, None
            except Exception as e2:
                return None, f"{e1}; {e2}"

def iso_one_decimal(dt):
    if dt is None:
        return ""
    if isinstance(dt, str):
        try:
            dt = dtparse.parse(dt)
        except Exception:
            return ""
    if not isinstance(dt, datetime):
        return ""
    factor = 100000  # microseconds per 0.1s
    total_us = dt.microsecond
    rounded_tenths = int(round(total_us / factor))
    new_micro = rounded_tenths * factor
    carry_s = new_micro // 1_000_000
    new_micro = new_micro % 1_000_000
    if carry_s:
        dt = dt.replace(microsecond=0) + timedelta(seconds=carry_s)
    else:
        dt = dt.replace(microsecond=new_micro)
    tenth_digit = dt.microsecond // factor
    frac = f".{tenth_digit:d}"
    if dt.tzinfo is None:
        return dt.strftime("%Y-%m-%dT%H:%M:%S") + frac
    else:
        base = dt.strftime("%Y-%m-%dT%H:%M:%S")
        offset = dt.strftime("%z")
        if offset:
            sign = offset[0]; hh = offset[1:3]; mm = offset[3:5]; tzs = f"{sign}{hh}:{mm}"
        else:
            tzs = "+00:00"
        return base + frac + tzs

def parse_header_times_and_exptime(hdr):
    if hdr is None:
        return None, None, ""
    utc_dt = None
    local_dt = None
    exposure_seconds = ""
    date_hdr = hdr.get("DATE")
    if date_hdr:
        try:
            dt = dtparse.parse(str(date_hdr))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            utc_dt = dt.astimezone(timezone.utc)
        except Exception:
            utc_dt = None
    date_obs = hdr.get("DATE-OBS")
    if date_obs:
        s = str(date_obs).strip()
        m = re.match(r"(?P<dt>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?)", s)
        iso_like = m.group("dt") if m else s
        try:
            dtl = dtparse.parse(iso_like)
            if dtl.tzinfo is None and LOCAL_TZ is not None:
                dtl = dtl.replace(tzinfo=LOCAL_TZ)
            local_dt = dtl
            if utc_dt is None:
                try:
                    utc_dt = local_dt.astimezone(timezone.utc)
                except Exception:
                    pass
        except Exception:
            local_dt = None
    expt = None
    for k in EXPT_KEYS:
        if k in hdr:
            expt = hdr.get(k); break
    if expt is not None:
        try:
            exposure_seconds = float(expt) * 1e-5
        except Exception:
            exposure_seconds = ""
    return utc_dt, local_dt, exposure_seconds

def gather_fits(root_dir):
    rows = []
    errors = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if not fn.lower().endswith((".fits", ".fit", ".fts")):
                continue
            fpath = os.path.join(dirpath, fn)
            obj, filt, date_str = parse_filename_info(fn)
            hdr, err = safe_open_fits(fpath)
            if err:
                errors.append((fpath, err))
            utc_dt, local_dt, exposure_seconds = parse_header_times_and_exptime(hdr)
            date_col = date_str or (local_dt.strftime("%Y-%m-%d") if local_dt else (utc_dt.strftime("%Y-%m-%d") if utc_dt else ""))
            utc_iso = iso_one_decimal(utc_dt) if utc_dt else ""
            local_iso = iso_one_decimal(local_dt) if local_dt else ""
            sort_dt = None
            if utc_dt:
                sort_dt = utc_dt
            elif local_dt:
                try:
                    sort_dt = local_dt.astimezone(timezone.utc)
                except Exception:
                    sort_dt = None
            if sort_dt is None:
                try:
                    mtime = os.path.getmtime(fpath)
                    sort_dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
                    if not utc_iso:
                        utc_iso = iso_one_decimal(sort_dt)
                    if not date_col:
                        date_col = sort_dt.strftime("%Y-%m-%d")
                except Exception:
                    sort_dt = datetime(1970,1,1, tzinfo=timezone.utc)
            rows.append({
                "file_path": fpath,
                "object_name": obj or "",
                "filter": filt or "",
                "date": date_col,
                "time_utc": utc_iso,
                "time_local": local_iso,
                "exposure_seconds": exposure_seconds if exposure_seconds != "" else "",
                "sort_dt": sort_dt
            })
    return rows, errors

def write_csv(rows, out_csv):
    rows_sorted = sorted(rows, key=lambda r: r["sort_dt"] or datetime(1970,1,1, tzinfo=timezone.utc))
    fieldnames = ["file_path","object_name","filter","date","time_utc","time_local","exposure_seconds"]
    out_path = Path(out_csv); out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames); writer.writeheader()
        for r in rows_sorted:
            out_row = {k: r.get(k, "") for k in fieldnames}
            es = out_row["exposure_seconds"]
            if isinstance(es, float):
                out_row["exposure_seconds"] = f"{es:.6f}"
            writer.writerow(out_row)
    return str(out_path), rows_sorted

def make_summary(rows_sorted, errors):
    total = len(rows_sorted)
    summary_lines = []
    summary_lines.append(f"Total files: {total}")
    if total == 0:
        summary_lines.append("No files to summarize.")
        return "\n".join(summary_lines)
    times_utc = [r["sort_dt"] for r in rows_sorted if r.get("sort_dt")]
    tmin = min(times_utc); tmax = max(times_utc)
    summary_lines.append(f"Overall time range (UTC): {iso_one_decimal(tmin)}  ->  {iso_one_decimal(tmax)}")
    # per-object
    per_obj = defaultdict(list)
    per_filter = Counter()
    exposures = []
    for r in rows_sorted:
        per_obj[r["object_name"]].append(r)
        per_filter[r["filter"]] += 1
        es = r.get("exposure_seconds")
        if isinstance(es, float):
            exposures.append(es)
        elif isinstance(es, str) and es != "":
            try:
                exposures.append(float(es))
            except Exception:
                pass
    summary_lines.append("")
    summary_lines.append("Per-object summary:")
    for obj, items in sorted(per_obj.items(), key=lambda x: (-len(x[1]), x[0])):
        count = len(items)
        times = [it["sort_dt"] for it in items if it.get("sort_dt")]
        if times:
            t0 = min(times); t1 = max(times)
            summary_lines.append(f"- {obj or '(unknown)'}: {count} files, time range {iso_one_decimal(t0)} -> {iso_one_decimal(t1)}")
        else:
            summary_lines.append(f"- {obj or '(unknown)'}: {count} files, no times")
    summary_lines.append("")
    summary_lines.append("Per-filter counts:")
    for filt, cnt in per_filter.most_common():
        summary_lines.append(f"- {filt or '(unknown)'}: {cnt}")
    summary_lines.append("")
    if exposures:
        summary_lines.append("Exposure time (s) stats:")
        summary_lines.append(f"- min: {min(exposures):.6f}")
        summary_lines.append(f"- mean: {statistics.mean(exposures):.6f}")
        summary_lines.append(f"- median: {statistics.median(exposures):.6f}")
        summary_lines.append(f"- max: {max(exposures):.6f}")
    else:
        summary_lines.append("No valid exposure times found.")
    if errors:
        summary_lines.append("")
        summary_lines.append(f"Files with header open errors: {len(errors)} (logged)")
    return "\n".join(summary_lines)

def write_summary(text, out_path):
    p = Path(out_path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return str(p)

def main():
    rows, errors = gather_fits(ROOT_DIR)
    csv_path, rows_sorted = write_csv(rows, OUT_CSV)
    summary_text = make_summary(rows_sorted, errors)
    summary_path = write_summary(summary_text, OUT_SUMMARY)
    print(summary_text)
    print()
    print(f"CSV written to: {csv_path}")
    print(f"Summary written to: {summary_path}")
    if errors:
        print(f"{len(errors)} files had header read errors (listed below):")
        for p,e in errors[:50]:
            print("-", p, "->", e)

if __name__ == "__main__":
    main()
