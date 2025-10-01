#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════#
#This code was written by Aysan Hemmatiortakand. Last updated 10/01/2025 
#you can contact me for any additional questions or information via Email 
#email address :aysanhemmatiortakand@gmail.com
#github = https://github.com/Aysikal
#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════#


# ─── 0) IERS Workaround & Warnings ─────────────────────────────────────────
from astropy.utils import iers
iers.conf.auto_max_age = None

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*predictive values that are more than.*",
    module="astropy.utils.iers"
)

# ─── 1) Core Imports ────────────────────────────────────────────────────────
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from astroplan import Observer

# ─── 2) Observatory Configuration ───────────────────────────────────────────
SITE_NAME     = "Iran National Observatory"
SITE_LAT      = 33.674       # degrees
SITE_LON      = 51.3188      # degrees
SITE_ELEV     = 3600         # meters
SITE_TIMEZONE = "Asia/Tehran"        # you can switch to "UTC" if you prefer universal time

SITE_LOCATION = EarthLocation(
    lat=SITE_LAT * u.deg,
    lon=SITE_LON * u.deg,
    height=SITE_ELEV * u.m
)
SITE_OBSERVER = Observer(
    location=SITE_LOCATION,
    timezone=SITE_TIMEZONE,
    name=SITE_NAME
)

# Visibility Plot Function ────────────────────────────────────────────
def visibility_plot(
    star_coords,
    date,
    n_steps=200,
    elevation_limit=30*u.deg,
    overlay=False
):
    """
    Plot altitude vs. time for one or more stars at the SITE.
    
    Parameters
    ----------
    star_coords : dict
      {"Name": (ra, dec), …}  RA/Dec as "HH:MM:SS"/"DD:MM:SS" or decimal degrees.
    date : str
      "YYYY-MM-DD" (UTC date of observation).
    n_steps : int
      Number of time samples between sunset and sunrise.
    elevation_limit : Quantity
      Draw a horizontal reference line at this elevation.
    overlay : bool
      True = all stars on one figure; False = one figure per star.
    """
    # Build the night‐time grid
    mid_day = Time(f"{date} 12:00:00", scale="utc")
    sunset  = SITE_OBSERVER.sun_set_time(mid_day, which="nearest")
    sunrise = SITE_OBSERVER.sun_rise_time(mid_day, which="next")
    total_hr = (sunrise - sunset).to(u.hour).value
    times = sunset + np.linspace(0, total_hr, n_steps) * u.hour

    # Prepare overlaid figure if requested
    if overlay:
        plt.figure(figsize=(10, 6))

    # Loop through each star
    for name, (ra, dec) in star_coords.items():
        unit = (u.hourangle, u.deg) if isinstance(ra, str) else (u.deg, u.deg)
        coord = SkyCoord(ra=ra, dec=dec, unit=unit, frame="icrs")
        altaz = coord.transform_to(
            AltAz(obstime=times, location=SITE_LOCATION)
        )

        if not overlay:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot_date(times.plot_date, altaz.alt.deg, "-")
            ax.set_title(f"{name} Altitude vs Time on {date}")
        else:
            plt.plot_date(times.plot_date, altaz.alt.deg, "-", label=name)
            ax = plt.gca()

        # Common formatting
        ax.axhline(elevation_limit.to_value(u.deg),
                   color="gray", ls="--",
                   label=f"{elevation_limit} Limit")
        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%H:%M", tz=SITE_TIMEZONE)
        )
        ax.set_xlabel("Time (" + SITE_TIMEZONE + ")")
        ax.set_ylabel("Altitude (°)")
        ax.set_ylim(0, 90)
        ax.grid(True)
        ax.legend(loc="lower right")

        if not overlay:
            plt.tight_layout()
            plt.show()

    if overlay:
        plt.title(f"All Targets on {date} at {SITE_NAME}")
        plt.tight_layout()
        plt.show()


# Airmass Function ─────────────────────────────────────────────────────
def airmass_function(
    date_str,
    hour,
    minute,
    RA,
    DEC,
    plot_night_curve=False,
    n_steps=200
):
    """
    Compute single‐time airmass and optionally overplot full‐night
    altitude-colored‐by-airmass at the SITE.
    
    Parameters
    ----------
    date_str : str
      "YYYY-MM-DD" (UTC)
    hour, minute : int
      UTC time for the single airmass calculation.
    RA, DEC : str
      "HH:MM:SS", "+DD:MM:SS"
    plot_night_curve : bool
      True = draw altitude vs time colored by airmass (1–3).
    n_steps : int
      Time resolution for the full‐night plot.
    """
    # Parse RA/Dec → decimal degrees
    ra_h, ra_m, ra_s = map(float, RA.split(":"))
    ra_deg = ra_h * 15 + ra_m * 0.25 + ra_s * (0.25/60)
    d, m, s = map(float, DEC.split(":"))
    sign = 1 if d >= 0 else -1
    dec_deg = sign * (abs(d) + m/60 + s/3600)

    # Single‐time airmass
    coord = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame="icrs")
    ts = f"{date_str} {hour:02d}:{minute:02d}:00"
    t0 = Time(ts, scale="utc")
    alt0 = coord.transform_to(
        AltAz(obstime=t0, location=SITE_LOCATION)
    ).alt.degree

    print(f"Altitude at {ts} UTC: {alt0:.2f}°")
    z0 = np.radians(90 - alt0)
    X0 = 1.0 / (
        np.cos(z0)
        + 0.50572 * (6.07995 + np.degrees(z0)) ** (-1.6364)
    )
    print(f"Airmass at {ts} UTC: {X0:.3f}")

    # Optional full‐night airmass plot
    if plot_night_curve:
        # reuse visibility_plot's time grid logic
        mid_day = Time(f"{date_str} 12:00:00", scale="utc")
        sunset  = SITE_OBSERVER.sun_set_time(mid_day, which="nearest")
        sunrise = SITE_OBSERVER.sun_rise_time(mid_day, which="next")
        total_hr = (sunrise - sunset).to(u.hour).value
        times = sunset + np.linspace(0, total_hr, n_steps) * u.hour

        # altitude & airmass arrays
        altaz = coord.transform_to(
            AltAz(obstime=times, location=SITE_LOCATION)
        )
        alts = altaz.alt.degree
        zs   = 90 - alts
        zr   = np.radians(zs)
        Xs   = 1.0 / (
            np.cos(zr)
            + 0.50572 * (6.07995 + zs) ** (-1.6364)
        )

        # cap and normalize to [1,3]
        norm = plt.Normalize(vmin=1, vmax=3)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot_date(times.plot_date, alts, "-k", label="Altitude")
        sc = ax.scatter(
            times.plot_date,
            alts,
            c=Xs,
            cmap="coolwarm",  # blue=low, red=high
            norm=norm,
            s=25,
            alpha=0.8
        )

        ax.axhline(30, color="gray", ls="--", label="30° Elevation")
        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%H:%M", tz=SITE_TIMEZONE)
        )
        ax.set_xlabel("Time (" + SITE_TIMEZONE + ")")
        ax.set_ylabel("Altitude (°)")
        ax.set_ylim(0, 90)
        ax.grid(True)

        cbar = plt.colorbar(sc, ax=ax, pad=0.02, extend="max")
        cbar.set_label("Airmass (clamped at 3)")

        ax.legend(loc="lower right")
        plt.title(f"{RA} {DEC} — Altitude & Airmass on {date_str} (UTC)")
        plt.tight_layout()
        plt.show()

    return X0