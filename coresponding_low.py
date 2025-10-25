import os
import shutil
import re

# ---------------- CONFIG ----------------
HIGH_KEEP = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\Oct01\oct01_2025\target2\i\high\keep"
LOW_FOLDER = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\Oct01\oct01_2025\target2\i\low"
LOW_KEEP = os.path.join(LOW_FOLDER, "keep")
os.makedirs(LOW_KEEP, exist_ok=True)

# ---------------- Helpers ----------------
def list_fits(folder):
    """Return sorted list of .fit or .fits files in folder"""
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(('.fit', '.fits'))])

def rename_high_to_low(high_name):
    """Convert high filename to corresponding low filename"""
    low_name = re.sub(r'_High_', '_low_', high_name, flags=re.IGNORECASE)
    return low_name

# ---------------- Main Logic ----------------
high_files = list_fits(HIGH_KEEP)
low_files = list_fits(LOW_FOLDER)

print(f"Found {len(high_files)} high-keep files, {len(low_files)} low files")

for hf in high_files:
    # Convert high name to expected low name
    low_name_expected = rename_high_to_low(hf)
    
    # Check if the low file exists in low folder
    low_path = os.path.join(LOW_FOLDER, low_name_expected)
    if os.path.exists(low_path):
        # Move low file to low/keep
        shutil.move(low_path, os.path.join(LOW_KEEP, low_name_expected))
        print(f"Moved {low_name_expected} to {LOW_KEEP}")
    else:
        print(f"⚠️ Low file {low_name_expected} not found, skipping.")

# ---------------- Summary ----------------
num_high = len(list_fits(HIGH_KEEP))
num_low_keep = len(list_fits(LOW_KEEP))
print(f"High keep files: {num_high}, Low keep files: {num_low_keep}")
