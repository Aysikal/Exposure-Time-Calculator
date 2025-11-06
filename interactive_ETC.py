import tkinter as tk
from tkinter import simpledialog, messagebox
import sys
import numpy as np
import math
from functions import airmass_function , get_fli , calculate_sky_magnitude

# Create the main window
root = tk.Tk()
root.title("Calculator")
root.geometry("400x250")
root.configure(bg='#F0F0F0')  # Light grey background

# Set custom font
custom_font = ('Helvetica', 12)

# Variables to store user inputs
mode = ''
year = month = day = hour = minute = None
RA = DEC = filter_choice = None
magnitude = None
binning = None
seeing_conditions = None
full_well = 70000  # System parameter

# Fixed values
pixel_scale = 0.047 * 1.8
dc = 0.08  # Dark current
h = 6.62620 * 10**(-34)  # Planck constant
c = 2.9979250 * 10**8  # Speed of light
readnoise = 3.7 #e
D = 3.4  # Telescope M1 diameter (m)
d = 0.6  # Telescope M2 diameter (m)
S = np.pi*(D/2)**2 - np.pi*(d/2)**2  # Mirror effective area (m^2)

# Effective wavelengths (Central Wavelength) in meters
CW = {'u': 3560e-10,
      'g': 4825e-10,
      'r': 6261e-10,
      'i': 7672e-10}

# Bandwidths in meters (Fukugita et al. 1996)
band_width = {'u': 463e-10,
              'g': 988e-10,
              'r': 1340e-10,
              'i': 1064e-10}

# Extinction values (should be updated based on actual data)
extinction = {'u': 0.404,
              'g': 0.35,
              'r': 0.2,
              'i': 0.15}

mirror_reflectivity = 0.7


eta = {'u': 0.0025,
       'g': 0.393,
       'r': 0.326,
       'i': 0.303}

E = {'u': 0.1 * (mirror_reflectivity)**2 * mirror_reflectivity,
     'g': 0.7 * (mirror_reflectivity)**2 * mirror_reflectivity,
     'r': 0.75 * (mirror_reflectivity)**2 * mirror_reflectivity,
     'i': 0.55 * (mirror_reflectivity)**2 * mirror_reflectivity,}

# Sky background magnitude when no moon is present
offset = {'u': 22,
          'g': 22,
          'r': 22,
          'i': 22,
          'z': 22}

# Base dialog class with custom font and colors
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
        sys.exit()  # User cancelled or closed the dialog, exit the program
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

# Function to ask for object inputs (RA, DEC, filter, magnitude)
def ask_object_inputs():
    global RA, DEC, filter_choice, magnitude
    messagebox.showinfo("Object Inputs", "Please enter the object details.")

    RA = ask_input("RA Input", prompt="Enter RA (HH:MM:SS):")
    DEC = ask_input("DEC Input", prompt="Enter DEC (DD:MM:SS):")
    filter_options = ["u", "g", "r", "i", "z"]  # Adjusted to match keys in dictionaries
    filter_choice = ask_input("Filter Selection", prompt="Choose a filter:", options=filter_options)
    messagebox.showinfo("Magnitude Input", "Enter magnitude in the chosen filter.\nNote: The magnitude should be in the AB system.")
    magnitude = ask_input("Magnitude Input", prompt="Enter magnitude:", input_type='float')

# Function to ask for system inputs (binning, seeing conditions)
def ask_system_inputs():
    global binning, seeing_conditions, seeing
    messagebox.showinfo("System Inputs", "Please enter the system details.")

    binning_options = ["1", "2"]  # Represents 1x1 and 2x2
    binning_choice = ask_input("Binning Selection", prompt="Choose binning (1x1 or 2x2):", options=binning_options)
    binning = int(binning_choice)

    seeing_options = ["optimal (0.6 - 0.8 arcseconds)", "minimal (0.8 - 1 arcseconds)",
                      "moderate (1 - 1.3 arcseconds)", "high (1.3 - 1.5 arcseconds)",
                      "very high (1.5 - 2 arcseconds)"]
    seeing_dict = {
        "optimal (0.6 - 0.8 arcseconds)": 0.7,     # Average of the range
        "minimal (0.8 - 1 arcseconds)": 0.9,
        "moderate (1 - 1.3 arcseconds)": 1.15,
        "high (1.3 - 1.5 arcseconds)": 1.4,
        "very high (1.5 - 2 arcseconds)": 1.75
    }
    seeing_conditions = ask_input("Seeing Conditions", prompt="Choose seeing conditions:", options=seeing_options)
    seeing = seeing_dict[seeing_conditions]

# Function to ask for date and time without confirmation
def ask_date_time():
    global year, month, day, hour, minute
    messagebox.showinfo("Attention", "Time and date entries MUST be UTC")
    year = ask_input("Date and Time", prompt="Enter the year (YYYY):", input_type='int')
    month = ask_input("Date and Time", prompt="Enter the month (1-12):", input_type='int')
    day = ask_input("Date and Time", prompt="Enter the day (1-31):", input_type='int')
    hour = ask_input("Date and Time", prompt="Enter the hour (0-23):", input_type='int')
    minute = ask_input("Date and Time", prompt="Enter the minute (0-59):", input_type='int')

# Function to calculate SNR
def calculate_snr(year, month, day, hour, minute, RA, DEC, seeing, pixel_scale, binning, h, c, CW, filter_choice, magnitude, extinction, band_width, exposure_time, E, S, get_fli, offset, calculate_sky_magnitude, readnoise):
    # Signal calculation
    airmass = airmass_function(year, month, day, hour, minute, RA, DEC)
    npix = (np.pi * ((seeing / pixel_scale) ** 2)) / (binning ** 2)
    P = (h * c) / CW[filter_choice]
    m_corrected = magnitude + (airmass * extinction[filter_choice])
    f_nu = 10 ** (-0.4 * (m_corrected + 48.6))
    f_lambda = (f_nu * c) / (CW[filter_choice] ** 2) * 1e-10  # erg/cm2/A
    A = (f_lambda * 1e-7 * (band_width[filter_choice] * 1e10) * E[filter_choice] * S * 1e4) / P
    signal = A * exposure_time

    # Sky calculation
    fli = get_fli(year, month, day, hour, minute)
    sky_mag = calculate_sky_magnitude(offset[filter_choice], fli)
    f_nu_s = 10 ** (-0.4 * (sky_mag + 48.6))
    f_lambda_s = (f_nu_s * c) / (CW[filter_choice] ** 2) * 1e-10  # erg/cm2/A
    C = (f_lambda_s * 1e-7 * (band_width[filter_choice] * 1e10) * E[filter_choice] * S * 1e4 * (pixel_scale ** 2)) / P
    N_sky = C * exposure_time

    # Noise calculation
    B = npix * (N_sky + readnoise ** 2)
    noise = np.sqrt(A * exposure_time + B)
    signal_to_noise = signal / noise
    return signal_to_noise

# Function to solve for exposure time
def solve_for_t(A, npix, C, readnoise, s):
    a = A**2
    b = -s**2 * (A + npix * C)
    c = -s**2 * npix * readnoise**2

    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        return None  # No real solution

    t1 = (-b + math.sqrt(discriminant)) / (2 * a)
    t2 = (-b - math.sqrt(discriminant)) / (2 * a)
    if t1 >= 0 and t2 >= 0:
        return min(t1, t2)
    elif t1 >= 0:
        return t1
    elif t2 >= 0:
        return t2
    else:
        return None  # No valid solution

# Function to calculate exposure time
def calculate_exposure_time(snr_value, year, month, day, hour, minute, RA, DEC, seeing, pixel_scale, binning, h, c, CW, filter_choice, magnitude, extinction, band_width, E, S, get_fli, offset, calculate_sky_magnitude, readnoise):
    airmass = airmass_function(year, month, day, hour, minute, RA, DEC)
    npix = (np.pi * ((seeing / pixel_scale) ** 2)) / (binning ** 2)
    P = (h * c) / CW[filter_choice]
    m_corrected = magnitude + (airmass * extinction[filter_choice])
    f_nu = 10 ** (-0.4 * (m_corrected + 48.6))
    f_lambda = (f_nu * c) / (CW[filter_choice] ** 2) * 1e-10  # erg/cm2/A
    A = (f_lambda * 1e-7 * (band_width[filter_choice] * 1e10) * E[filter_choice] * S * 1e4) / P

    fli = get_fli(year, month, day, hour, minute)
    sky_mag = calculate_sky_magnitude(offset[filter_choice], fli)
    f_nu_s = 10 ** (-0.4 * (sky_mag + 48.6))
    f_lambda_s = (f_nu_s * c) / (CW[filter_choice] ** 2) * 1e-10  # erg/cm2/A
    C = (f_lambda_s * 1e-7 * (band_width[filter_choice] * 1e10) * E[filter_choice] * S * 1e4 * (pixel_scale ** 2)) / P

    exposure_time = solve_for_t(A, npix, C, readnoise, snr_value)
    return exposure_time

# Function for SNR calculator
def snr_calculator():
    global mode
    mode = 'snr'
    root.withdraw()  # Hide the main window instead of destroying it
    ask_date_time()
    ask_object_inputs()
    ask_system_inputs()
    messagebox.showinfo("Attention", "Exposure time should be in seconds")
    while True:
        exposure_time = ask_input("SNR Calculator", prompt="Enter exposure time (seconds):", input_type='float')
        message = f"Entered Exposure Time:\n{exposure_time} seconds\n\nIs this correct?"
        confirm = messagebox.askyesno("Confirm Exposure Time", message)
        if confirm:
            messagebox.showinfo("SNR Calculator", f"Exposure time confirmed:\n{exposure_time} seconds")
            # Proceed with calculations using collected data
            process_snr_calculation(exposure_time)
            break
        else:
            continue

# Function for Exposure Time calculator
def exp_calculator():
    global mode
    mode = 'exp'
    root.withdraw()  # Hide the main window instead of destroying it
    ask_date_time()
    ask_object_inputs()
    ask_system_inputs()
    messagebox.showinfo("Attention", "Enter desired SNR value")
    while True:
        snr_value = ask_input("Exposure Time Calculator", prompt="Enter desired SNR value:", input_type='float')
        message = f"Entered SNR Value:\n{snr_value}\n\nIs this correct?"
        confirm = messagebox.askyesno("Confirm SNR Value", message)
        if confirm:
            messagebox.showinfo("Exposure Time Calculator", f"SNR value confirmed:\n{snr_value}")
            # Proceed with calculations using collected data
            process_exposure_time_calculation(snr_value)
            break
        else:
            continue

# Processing functions
def process_snr_calculation(exposure_time):
    global mode, year, month, day, hour, minute, RA, DEC, magnitude, filter_choice, binning, seeing_conditions, seeing
    # Calculate SNR
    snr = calculate_snr(year, month, day, hour, minute, RA, DEC, seeing, pixel_scale, binning, h, c, CW,
                        filter_choice, magnitude, extinction, band_width, exposure_time, E, S,
                        get_fli, offset, calculate_sky_magnitude, readnoise)

    # Print all variables and results
    result_message = f"""Calculating SNR with the following inputs:

Mode: {mode}
Date and Time (UTC): {year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}
RA: {RA}
DEC: {DEC}
Filter: {filter_choice}
Magnitude: {magnitude}
Binning: {binning}x{binning}
Seeing Conditions: {seeing_conditions}
Exposure Time: {exposure_time} seconds

Calculated SNR: {snr:.2f}
"""
    print(result_message)  # Print to console
    messagebox.showinfo("SNR Calculation Result", result_message)
    sys.exit()  # Terminate after displaying results

def process_exposure_time_calculation(snr_value):
    global mode, year, month, day, hour, minute, RA, DEC, magnitude, filter_choice, binning, seeing_conditions, seeing
    # Calculate Exposure Time
    exposure_time = calculate_exposure_time(snr_value, year, month, day, hour, minute, RA, DEC, seeing,
                                            pixel_scale, binning, h, c, CW, filter_choice, magnitude, extinction,
                                            band_width, E, S, get_fli, offset, calculate_sky_magnitude, readnoise)
    if exposure_time is None:
        result_message = "Unable to calculate exposure time with given parameters."
    else:
        result_message = f"""Calculating Exposure Time with the following inputs:

Mode: {mode}
Date and Time (UTC): {year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}
RA: {RA}
DEC: {DEC}
Filter: {filter_choice}
Magnitude: {magnitude}
Binning: {binning}x{binning}
Seeing Conditions: {seeing_conditions}
Desired SNR: {snr_value}

Calculated Exposure Time: {exposure_time:.2f} seconds
"""
    print(result_message)  # Print to console
    messagebox.showinfo("Exposure Time Calculation Result", result_message)
    sys.exit()  # Terminate after displaying results

# Style the main window buttons
button_style = {
    'font': custom_font,
    'bg': '#008CBA',
    'fg': 'white',
    'activebackground': '#007BA7',
    'activeforeground': 'white',
    'width': 30,
    'bd': 0,
    'cursor': 'hand2',
}

# Create and place buttons in the main window
snr_button = tk.Button(root, text="SNR Calculator", command=snr_calculator, **button_style)
snr_button.pack(pady=15)

exp_button = tk.Button(root, text="Exposure Time Calculator", command=exp_calculator, **button_style)
exp_button.pack()

# Run the main loop
root.mainloop()


#CURRENT BEST