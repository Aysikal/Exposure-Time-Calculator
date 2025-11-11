import tkinter as tk
from tkinter import simpledialog, messagebox
import sys
import numpy as np
import math
from datetime import datetime
from zoneinfo import ZoneInfo
from astropy.time import Time
from astropy.coordinates import SkyCoord, AltAz, EarthLocation
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from ancillary_functions import airmass_function, get_fli, calculate_sky_magnitude
# -------------------------------------------------------------------
# Site configuration
# -------------------------------------------------------------------
SITE_NAME      = "INO"
SITE_LAT, SITE_LON, SITE_ELEV = 35.674, 51.3188, 3600
SITE_LOCATION  = EarthLocation(lat=SITE_LAT*u.deg,
                               lon=SITE_LON*u.deg,
                               height=SITE_ELEV*u.m)
LOCAL_TIMEZONE   = "Asia/Tehran"


# -------------------------
# Main Window Setup
# -------------------------
root = tk.Tk()
root.title("Calculator")
root.geometry("400x250")
root.configure(bg='#F0F0F0')

custom_font = ('Helvetica', 12)

# -------------------------
# Variables
# -------------------------
mode = ''
year = month = day = hour = minute = None
RA = DEC = filter_choice = None
magnitude = None
binning = None
seeing_conditions = None
non_linearity_error = 3500 #ADU
over_exposure_error = 4000 #ADU 
pixel_scale = 0.101 
dc = 0.08
h = 6.62620e-34
c = 2.9979250e8
readnoise = 3.7  # electrons
D = 3.4
d = 0.6
S = np.pi*(D/2)**2 - np.pi*(d/2)**2  # m^2
S_cm2 = S * 1e4  # cm^2

# -------------------------
# Gain (ADU -> electrons)
# -------------------------
GAIN = 1  

# -------------------------
# Filters, Bandwidths, Extinction
# -------------------------
# Original numbers kept but treated explicitly as meters.
# e.g., 6261e-10 m = 626.1 nm = 6.261e-7 m
CW = {'u': 3540e-10, 'g': 4770e-10, 'r': 6230e-10, 'i': 7630e-10}        # central wavelength in meters
band_width = {'u': 600e-10, 'g': 1380e-10, 'r': 1380e-10, 'i': 1520e-10} # effective bandwidth in meters
extinction = {'u': 0.404, 'g': 0.35, 'r': 0.2, 'i': 0.15}  

# -------------------------
# System Efficiency (fraction)
# -------------------------
E = 0.11

# -------------------------
# GUI & Input functions
# -------------------------
class BaseDialog(simpledialog.Dialog):
    def __init__(self, parent, title=None, message=None, prompt=None, options=None, initialvalue='', input_type='string'):
        self.message = message
        self.prompt = prompt
        self.options = options
        self.initialvalue = initialvalue
        self.input_type = input_type
        self.result = None
        super().__init__(parent, title=title)

    def body(self, master):
        master.configure(bg='#FFFFFF')
        if self.message:
            tk.Label(master, text=self.message, font=custom_font, bg='#FFFFFF').pack(pady=10, padx=10)
        if self.prompt and self.options:
            self.var = tk.StringVar(value=self.options[0])
            tk.Label(master, text=self.prompt, font=custom_font, bg='#FFFFFF').pack(pady=10)
            for option in self.options:
                tk.Radiobutton(master, text=option, variable=self.var, value=option, font=custom_font,
                               bg='#FFFFFF', anchor='w').pack(pady=2, padx=20, anchor='w')
            return None
        if self.prompt:
            tk.Label(master, text=self.prompt, font=custom_font, bg='#FFFFFF').pack(pady=10)
            self.entry = tk.Entry(master, font=custom_font)
            self.entry.insert(0, self.initialvalue)
            self.entry.pack(pady=5, padx=10)
            return self.entry

    def buttonbox(self):
        box = tk.Frame(self, bg='#FFFFFF')
        tk.Button(box, text="OK", width=10, command=self.ok, font=custom_font,
                  bg='#4CAF50', fg='white', activebackground='#45A049').pack(side='left', padx=5, pady=5)
        tk.Button(box, text="Cancel", width=10, command=self.cancel, font=custom_font,
                  bg='#F44336', fg='white', activebackground='#E53935').pack(side='right', padx=5, pady=5)
        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)
        self.protocol("WM_DELETE_WINDOW", self.cancel)
        box.pack()

    def apply(self):
        if self.prompt and self.options:
            self.result = self.var.get()
        elif self.prompt:
            self.result = self.entry.get()

    def ok(self, event=None):
        self.apply()
        self.destroy()

    def cancel(self, event=None):
        self.result = None
        self.destroy()

def ask_input(title, message=None, prompt=None, options=None, initialvalue='', input_type='string'):
    dialog = BaseDialog(root, title=title, message=message, prompt=prompt,
                        options=options, initialvalue=initialvalue, input_type=input_type)
    if dialog.result is None:
        sys.exit()
    if input_type == 'float':
        try:
            return float(dialog.result)
        except ValueError:
            messagebox.showerror("Error", "Invalid input, please enter a valid number.")
            return ask_input(title, message, prompt, options, initialvalue, input_type)
    elif input_type == 'int':
        try:
            return int(dialog.result)
        except ValueError:
            messagebox.showerror("Error", "Invalid input, please enter a valid integer.")
            return ask_input(title, message, prompt, options, initialvalue, input_type)
    return dialog.result.strip()

def ask_object_inputs():
    global RA, DEC, filter_choice, magnitude
    messagebox.showinfo("Object Inputs", "Please enter the object details.")
    RA = ask_input("RA Input", prompt="Enter RA (HH:MM:SS):")
    DEC = ask_input("DEC Input", prompt="Enter DEC (DD:MM:SS):")
    filter_options = ["u", "g", "r", "i", "z"]
    filter_choice = ask_input("Filter Selection", prompt="Choose a filter:", options=filter_options)
    magnitude = ask_input("Magnitude Input", prompt="Enter magnitude:", input_type='float')

def ask_system_inputs():
    global binning, seeing_conditions, seeing
    binning_choice = ask_input("Binning Selection", prompt="Choose binning (1x1 or 2x2):", options=["1", "2"])
    binning = int(binning_choice)
    seeing_options = ["optimal (0.6 - 0.8 arcseconds)", "minimal (0.8 - 1 arcseconds)",
                      "moderate (1 - 1.3 arcseconds)", "high (1.3 - 1.5 arcseconds)",
                      "very high (1.5 - 2 arcseconds)"]
    seeing_dict = {"optimal (0.6 - 0.8 arcseconds)": 0.7,
                   "minimal (0.8 - 1 arcseconds)": 0.9,
                   "moderate (1 - 1.3 arcseconds)": 1.15,
                   "high (1.3 - 1.5 arcseconds)": 1.4,
                   "very high (1.5 - 2 arcseconds)": 1.75}
    seeing_conditions = ask_input("Seeing Conditions", prompt="Choose seeing conditions:", options=seeing_options)
    seeing = seeing_dict[seeing_conditions]

def ask_date_time():
    global year, month, day, hour, minute
    messagebox.showinfo("Attention", "Time and date entries MUST be local (Tehran time)")
    year = ask_input("Date and Time", prompt="Enter the year (YYYY):", input_type='int')
    month = ask_input("Date and Time", prompt="Enter the month (1-12):", input_type='int')
    day = ask_input("Date and Time", prompt="Enter the day (1-31):", input_type='int')
    hour = ask_input("Date and Time", prompt="Enter the hour (0-23):", input_type='int')
    minute = ask_input("Date and Time", prompt="Enter the minute (0-59):", input_type='int')

# -------------------------
# SNR & Exposure Calculations
# -------------------------
def calculate_snr(year, month, day, hour, minute, RA, DEC, seeing, pixel_scale, binning, h, c, CW, filter_choice,
                  magnitude, extinction, band_width, exposure_time, E, S, get_fli, calculate_sky_magnitude, readnoise, gain):

    # airmass
    date_str = f"{year}-{month:02d}-{day:02d}"
    airmass = airmass_function(date_str, hour, minute, RA, DEC)

    # aperture pixel count (keep your formula)
    npix = np.pi * ((seeing / pixel_scale*2*binning)** 2)

    # central wavelength and bandwidth in meters (already provided)
    wav_m = CW[filter_choice]
    bw_m = band_width[filter_choice]

    # AB flux density f_nu in erg s^-1 cm^-2 Hz^-1
    m_corrected = magnitude + (airmass * extinction[filter_choice])
    f_nu_erg = 10 ** (-0.4 * (m_corrected + 48.6))

    # convert to Joules: 1 erg = 1e-7 J
    f_nu_J = f_nu_erg * 1e-7  # J s^-1 cm^-2 Hz^-1

    # Convert to f_lambda in J s^-1 cm^-2 m^-1 : f_lambda = f_nu * c / lambda^2
    f_lambda_J_per_m = f_nu_J * c / (wav_m ** 2)  # J s^-1 cm^-2 m^-1

    # Apply atmospheric extinction (magnitudes) multiplicatively via flux ratio
    mag_atm = m_corrected
    flux_ratio = 10 ** (-0.4 * (mag_atm - magnitude))
    f_lambda_atm = f_lambda_J_per_m * flux_ratio  # J s^-1 cm^-2 m^-1 after atmosphere

    # Power per cm^2 collected across the band (J s^-1 cm^-2)
    power_per_cm2 = f_lambda_atm * bw_m

    # Total power on telescope collecting area (J / s)
    total_power_J_per_s = power_per_cm2 * (S * 1e4)  # S in m^2 -> S*1e4 cm^2

    # Photon energy at central wavelength (J)
    E_photon = h * c / wav_m

    # Photons per second collected by telescope (photons / s)
    photons_per_s = total_power_J_per_s / E_photon

    # Electrons per second produced by system (apply net efficiency E)
    electrons_per_s = photons_per_s * E

    # ADU per second (before/after gain depends on your convention); convert to electrons/sec directly:
    A_e_per_sec = electrons_per_s  # electrons / s from source in the aperture

    # signal electrons in exposure
    signal_e = A_e_per_sec * exposure_time

    # --- sky contribution ---
    fli = get_fli(date_str, hour, minute)
    sky_mag = calculate_sky_magnitude(date_str, hour, minute, RA, DEC)

    f_nu_s_erg = 10 ** (-0.4 * (sky_mag + 48.6))
    f_nu_s_J = f_nu_s_erg * 1e-7
    f_lambda_s_J_per_m = f_nu_s_J * c / (wav_m ** 2)
    # apply atmosphere for sky using same ext correction approach
    # (sky_mag assumed already represents observed sky magnitude; if it's top-of-atmosphere, include extinction)
    power_sky_per_cm2 = f_lambda_s_J_per_m * bw_m  # J s^-1 cm^-2 across band
    total_sky_power_J_per_s = power_sky_per_cm2 * (S * 1e4)  # J s^-1 on telescope
    photons_sky_per_s = total_sky_power_J_per_s / E_photon  # photons / s falling on telescope across the band
    # convert to electrons / s (system efficiency)
    electrons_sky_per_s = photons_sky_per_s * E

    # Now per-pixel sky: photons hitting telescope distribute over focal plane; convert using pixel scale
    # Pixel area on sky in arcsec^2: pixel_scale^2. Convert surface brightness to electrons per second per pixel:
    # The factor to go from total telescope-collected sky photons to per-pixel depends on how you define sky_mag (we follow original pattern):
    # Scale electrons_sky_per_s by pixel area fraction: pixel_area_arcsec2 / (FOV area normalization)
    # Here we mimic original pattern: compute per-pixel electrons using (pixel_scale^2) factor applied similarly to C in original code.
    # Simpler consistent approach: compute sky surface brightness power per arcsec^2, then multiply by pixel area.
    # To avoid changing overall scaling drastically, keep original style: compute C_e_per_sec_per_pix proportional to electrons_sky_per_s * (pixel_scale**2) * (1/SOME_NORMAL)
    # We'll compute a per-pixel electrons/sec by distributing telescope-collected photons over an approximate plate scale area:
    # The original code used: C = (f_lambda_s * 1e-7 * (band * 1e10) * E * S * 1e4 * (pixel_scale ** 2)) / P
    # Recreating consistent units: electrons_sky_per_s * (pixel_scale ** 2) divided by an approximate normalization solid angle.
    # For simplicity and unit consistency we compute per-pixel electrons by scaling the telescope-collected sky electrons by the pixel solid angle fraction:
    # pixel_area_sr is not known here; we use proportional factor pixel_scale**2 as in original code (keeps relative scaling).
    C_e_per_sec_per_pix = electrons_sky_per_s * (pixel_scale ** 2)  # electrons / s / pix (keeps your original scaling style)
    N_sky_e_per_pix = C_e_per_sec_per_pix * exposure_time

    # Total sky electrons inside aperture (npix is number of pixels in aperture)
    B = npix * (N_sky_e_per_pix + readnoise ** 2)
    noise = np.sqrt(max(signal_e, 0.0) + B)
    if noise <= 0:
        return 0.0
    return signal_e / noise

def solve_for_t(A_e_per_sec, npix, C_e_per_sec_per_pix, readnoise, s):
    # Solve quadratic for t using electrons/sec quantities:
    # SNR^2 = (A t)^2 / (A t + npix*(C t + rn^2))
    # Rearranged: (A^2) t^2 - s^2 (A + npix*C) t - s^2 npix rn^2 = 0
    a = A_e_per_sec ** 2
    b = -s**2 * (A_e_per_sec + npix * C_e_per_sec_per_pix)
    c = -s**2 * npix * (readnoise ** 2)
    disc = b**2 - 4*a*c
    if disc < 0:
        return None
    t1 = (-b + math.sqrt(disc)) / (2*a)
    t2 = (-b - math.sqrt(disc)) / (2*a)
    candidates = [t for t in (t1, t2) if t is not None and t >= 0]
    if not candidates:
        return None
    return min(candidates)

def calculate_exposure_time(snr_value, year, month, day, hour, minute, RA, DEC, seeing, pixel_scale, binning,
                            h, c, CW, filter_choice, magnitude, extinction, band_width, E, S, get_fli, offset,
                            calculate_sky_magnitude, readnoise, gain):

    date_str = f"{year}-{month:02d}-{day:02d}"
    airmass = airmass_function(date_str, hour, minute, RA, DEC)

    npix = np.pi * ((seeing / pixel_scale*2*binning)** 2)
    wav_m = CW[filter_choice]
    bw_m = band_width[filter_choice]

    # source photons/electrons per second (unit-consistent)
    m_corrected = magnitude + (airmass * extinction[filter_choice])
    f_nu_erg = 10 ** (-0.4 * (m_corrected + 48.6))
    f_nu_J = f_nu_erg * 1e-7
    f_lambda_J_per_m = f_nu_J * c / (wav_m ** 2)
    power_per_cm2 = f_lambda_J_per_m * bw_m
    total_power_J_per_s = power_per_cm2 * (S * 1e4)
    E_photon = h * c / wav_m
    photons_per_s = total_power_J_per_s / E_photon
    A_e_per_sec = photons_per_s * E

    # sky per-pixel electrons/sec (consistent with above)
    fli = get_fli(date_str, hour, minute)
    sky_mag = calculate_sky_magnitude(date_str, hour, minute, RA, DEC)
    f_nu_s_erg = 10 ** (-0.4 * (sky_mag + 48.6))
    f_nu_s_J = f_nu_s_erg * 1e-7
    f_lambda_s_J_per_m = f_nu_s_J * c / (wav_m ** 2)
    power_sky_per_cm2 = f_lambda_s_J_per_m * bw_m
    total_sky_power_J_per_s = power_sky_per_cm2 * (S * 1e4)
    photons_sky_per_s = total_sky_power_J_per_s / E_photon
    electrons_sky_per_s = photons_sky_per_s * E
    C_e_per_sec_per_pix = electrons_sky_per_s * (pixel_scale ** 2)

    return solve_for_t(A_e_per_sec, npix, C_e_per_sec_per_pix, readnoise, snr_value)

# -------------------------
# User Interaction
# -------------------------
def snr_calculator():
    global mode
    mode = 'snr'
    root.withdraw()
    ask_date_time()
    ask_object_inputs()
    ask_system_inputs()
    messagebox.showinfo("Attention", "Exposure time should be in seconds")
    while True:
        exposure_time = ask_input("SNR Calculator", prompt="Enter exposure time (seconds):", input_type='float')
        if messagebox.askyesno("Confirm Exposure Time", f"Entered Exposure Time:\n{exposure_time} seconds\n\nIs this correct?"):
            process_snr_calculation(exposure_time)
            break

def exp_calculator():
    global mode
    mode = 'exp'
    root.withdraw()
    ask_date_time()
    ask_object_inputs()
    ask_system_inputs()
    messagebox.showinfo("Attention", "Enter desired SNR value")
    while True:
        snr_value = ask_input("Exposure Time Calculator", prompt="Enter desired SNR value:", input_type='float')
        if messagebox.askyesno("Confirm SNR Value", f"Entered SNR Value:\n{snr_value}\n\nIs this correct?"):
            process_exposure_time_calculation(snr_value)
            break

def process_snr_calculation(exposure_time):
    snr = calculate_snr(year, month, day, hour, minute, RA, DEC, seeing, pixel_scale, binning,
                        h, c, CW, filter_choice, magnitude, extinction, band_width,
                        exposure_time, E, S, get_fli, calculate_sky_magnitude, readnoise, GAIN)
    message = f"Calculated SNR: {snr:.2f}"
    print(message)
    messagebox.showinfo("SNR Calculation Result", message)
    sys.exit()

def process_exposure_time_calculation(snr_value):
    exposure_time = calculate_exposure_time(snr_value, year, month, day, hour, minute, RA, DEC, seeing,
                                            pixel_scale, binning, h, c, CW, filter_choice, magnitude,
                                            extinction, band_width, E, S, get_fli, calculate_sky_magnitude,
                                            readnoise, GAIN)
    if exposure_time is None:
        message = "Unable to calculate exposure time with given parameters."
    else:
        message = f"Calculated Exposure Time: {exposure_time:.2f} seconds"
    print(message)
    messagebox.showinfo("Exposure Time Calculation Result", message)
    sys.exit()

# -------------------------
# GUI Buttons
# -------------------------
button_style = {'font': custom_font, 'bg': '#008CBA', 'fg': 'white',
                'activebackground': '#007BA7', 'activeforeground': 'white',
                'width': 30, 'bd': 0, 'cursor': 'hand2'}

tk.Button(root, text="SNR Calculator", command=snr_calculator, **button_style).pack(pady=15)
tk.Button(root, text="Exposure Time Calculator", command=exp_calculator, **button_style).pack()

root.mainloop()
