#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════#
#This code was written by Aysan Hemmatiortakand. Last updated 10/01/2025
#you can contact me for any additional questions or information via Email 
#email address :aysanhemmatiortakand@gmail.com
#github = https://github.com/Aysikal
#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════#

# 0) Work around IERS issues
from astropy.utils import iers
iers.conf.auto_max_age = None

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*predictive values that are more than.*",
    module="astropy.utils.iers"
)

# 1) Imports
from astropy.time import Time
from astropy.coordinates import SkyCoord, AltAz
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from astroplan import Observer

# 2) Define observer (INO) and date
ino = Observer(
    latitude=30.5 * u.deg,
    longitude=52.9 * u.deg,
    elevation=3170 * u.m,
    timezone="Asia/Tehran",
    name="Iran National Observatory"
)
date = "2025-09-30"
mid_day = Time(f"{date} 12:00:00")

# 3) Compute sunset and sunrise
sunset  = ino.sun_set_time(mid_day, which='nearest')
sunrise = ino.sun_rise_time(mid_day, which='next')

# 4) Build time grid
n_steps      = 200
duration_hr  = (sunrise - sunset).to(u.hour).value
delta_hours  = np.linspace(0, duration_hr, n_steps) * u.hour
times        = sunset + delta_hours


# 5) Your list of five stars (RA in HH:MM:SS, Dec in DD:MM:SS)
star_coords = {
    "BD+17 4708": ("22:11:31.4", "+18:05:34.1"),
    "BD+28 4211": ("21:51:11.0", "+28:51:50.4"),
    "BD+33 4737": ("23:34:36.1", "+34:05:36.6"),
    "BD+38 4955": ("23:13:38.8", "+39:25:02.6"),
    "BD+71 0031": ("00:43:44.3", "+72:10:47.3")
}
# 6) Plot each star individually
for name, (ra_hms, dec_dms) in star_coords.items():
    coord = SkyCoord(ra=ra_hms, dec=dec_dms, unit=(u.hourangle, u.deg), frame="icrs")
    altaz = coord.transform_to(AltAz(obstime=times, location=ino.location))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot_date(times.plot_date, altaz.alt.deg, "-", label=name)
    ax.axhline(30, color="gray", ls="--", label="30° Elevation")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=ino.timezone))
    ax.set_xlabel("Local Time (IRST)")
    ax.set_ylabel("Altitude (°)")
    ax.set_title(f"{name} Altitude vs Time on {date}")
    ax.set_ylim(0, 90)
    ax.grid(True)
    ax.legend(loc="lower right")
    fig.tight_layout()
    plt.show()

# 7) Combined overlay plot
plt.figure(figsize=(10, 6))
for name, (ra_hms, dec_dms) in star_coords.items():
    coord = SkyCoord(ra=ra_hms, dec=dec_dms, unit=(u.hourangle, u.deg), frame="icrs")
    altaz = coord.transform_to(AltAz(obstime=times, location=ino.location))
    plt.plot_date(times.plot_date, altaz.alt.deg, "-", label=name)

plt.axhline(30, color="gray", ls="--", label="30° Elevation Limit")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=ino.timezone))
plt.xlabel("Local Time (IRST)")
plt.ylabel("Altitude (°)")
plt.title(f"All Five BD Stars on {date} at INO")
plt.ylim(0, 90)
plt.grid(True)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()