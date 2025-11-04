import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# === Load two FITS files ===
file1 = r"C:\Users\AYSAN\Desktop\project\INO\ETC\Data\Oct01\oct01_2025\target2\g\high\keep\area92_g_T10C_2025_10_01_2x2_exp00.00.10.000_000001_High_1.fit"
file2 = r"C:\Users\AYSAN\Desktop\project\INO\ETC\area92_cycleclean_iter3_20251102T185444.fit"

data1 = fits.getdata(file1).astype(np.float32)
data2 = fits.getdata(file2).astype(np.float32)

# === Resolution function ===
def resolution(image):
    image_fourier = np.abs(np.fft.fftshift(np.fft.fft2(image)))
    max_radius = int(np.sqrt(image_fourier.shape[0]**2 + image_fourier.shape[1]**2) / 2)
    profile = np.zeros(max_radius + 1)
    count = np.zeros(max_radius + 1)
    for i in range(image_fourier.shape[0]):
        for j in range(image_fourier.shape[1]):
            r = int(np.sqrt((i - image_fourier.shape[0] / 2)**2 + (j - image_fourier.shape[1] / 2)**2))
            profile[r] += image_fourier[i, j]
            count[r] += 1
    return profile / np.maximum(count, 1)

# === Compute resolution profiles ===
res1 = resolution(data1)
res2 = resolution(data2)
diff = res1 - res2

# === Plot 1: Linear scale comparison ===
plt.figure(figsize=(10, 6))
plt.plot(res1, label='Raw Image', color='blue')
plt.plot(res2, label='Cleaned Image', color='orange', linestyle=':')
plt.xlabel("k 1/arcsecond")
plt.ylabel("Power")
plt.title("Resolution Comparison (Linear Scale)")
plt.xlim((0, 20))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Plot 2: Log scale comparison ===
plt.figure(figsize=(10, 6))
plt.semilogy(res1, label='Raw Image', color='blue')
plt.semilogy(res2, label='Cleaned Image', color='black', linestyle=':')
plt.xlabel("k (1/arcsecond)")
plt.ylabel("Log Power")
plt.title("Resolution Comparison (Log Scale)")
plt.xlim((0, 20))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Plot 3: Difference profile ===
plt.figure(figsize=(10, 6))
plt.plot(diff, label='Raw - Cleaned', color='green')
plt.xlabel("k (1/arcsecond)")
plt.ylabel("Difference Power")
plt.title("Difference in Resolution Profile")
plt.xlim((0, 20))
plt.axhline(0, color='gray', linestyle='--')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Plot 4: Fourier magnitude images ===
def show_fourier(image, title):
    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(image))))
    plt.figure(figsize=(6, 6))
    plt.imshow(spectrum, cmap='gray_r', origin='lower')
    plt.xlim((0, 20))
    plt.title(title)
    plt.colorbar(label='log(1 + |FFT|)')
    plt.tight_layout()
    plt.show()

show_fourier(data1, "Raw Image Fourier Spectrum")
show_fourier(data2, "Cleaned Image Fourier Spectrum")
