from astropy.utils import iers
iers.conf.auto_max_age = None

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*predictive values that are more than.*",
    module="astropy.utils.iers"
)

from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from astroplan import Observer

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# -----------------------------------------------------------------------------
# Site configuration
# -----------------------------------------------------------------------------
SITE_NAME      = "INO"
SITE_LAT, SITE_LON, SITE_ELEV = 35.674, 51.3188, 3600
SITE_LOCATION  = EarthLocation(lat=SITE_LAT*u.deg,
                               lon=SITE_LON*u.deg,
                               height=SITE_ELEV*u.m)
SITE_OBSERVER  = Observer(location=SITE_LOCATION,
                          timezone="UTC",
                          name=SITE_NAME)

# -----------------------------------------------------------------------------
# Optional custom time range for plotting (local time)
# -----------------------------------------------------------------------------
LOCAL_START_TIME = "2025-10-01 01:15"  # format: YYYY-MM-DD HH:MM
LOCAL_END_TIME   = "2025-10-01 03:58"
LOCAL_TIMEZONE   = "Asia/Tehran"

# -----------------------------------------------------------------------------
def airmass_function(
    date_str: str,
    hour: int,
    minute: int,
    RA: str,
    DEC: str,
    name: str,
    input_timezone: str = "UTC",
    plot_night_curve: bool = False,
    n_steps: int = 200,
) -> float:
    """
    Compute airmass at a given time (UTC or local) and optionally plot
    the full-night altitude curve color-mapped by airmass.
    """
    # 1) Build timezone objects
    tz_in  = ZoneInfo(input_timezone)
    tz_utc = ZoneInfo("UTC")

    # 2) Convert input date/time to UTC Time object
    dt_local = datetime.strptime(f"{date_str} {hour:02d}:{minute:02d}",
                                 "%Y-%m-%d %H:%M")
    dt_local = dt_local.replace(tzinfo=tz_in)
    dt_utc   = dt_local.astimezone(tz_utc)
    obs_time = Time(dt_utc)

    # 3) Parse RA/DEC and compute single-time airmass
    ra_h, ra_m, ra_s = map(float, RA.split(":"))
    ra_deg = ra_h*15 + ra_m*0.25 + ra_s*(0.25/60)
    d, m, s = map(float, DEC.split(":"))
    dec_sign = 1 if d >= 0 else -1
    dec_deg  = dec_sign*(abs(d) + m/60 + s/3600)

    coord  = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame="icrs")
    alt0   = coord.transform_to(
                AltAz(obstime=obs_time, location=SITE_LOCATION)
             ).alt.degree
    z0     = np.radians(90 - alt0)
    X0     = 1.0 / (
                np.cos(z0)
              + 0.50572*(6.07995 + np.degrees(z0))**(-1.6364)
             )

    print(f"Input time ({input_timezone}): {dt_local.strftime('%Y-%m-%d %H:%M')}")
    print(f"Converted UTC:     {dt_utc.strftime('%Y-%m-%d %H:%M')}")
    print(f"Altitude: {alt0:.2f}°,  Airmass: {X0:.3f}")

    # 4) Optional full-night plot
    if plot_night_curve:
        # Convert local start/end to UTC Time objects
        tz_local = ZoneInfo(LOCAL_TIMEZONE)
        start_dt = datetime.strptime(LOCAL_START_TIME, "%Y-%m-%d %H:%M").replace(tzinfo=tz_local)
        end_dt   = datetime.strptime(LOCAL_END_TIME, "%Y-%m-%d %H:%M").replace(tzinfo=tz_local)
        start_utc = Time(start_dt.astimezone(ZoneInfo("UTC")))
        end_utc   = Time(end_dt.astimezone(ZoneInfo("UTC")))

        # Build time grid
        hours = (end_utc - start_utc).to(u.hour).value
        times = start_utc + np.linspace(0, hours, n_steps)*u.hour

        # Compute altitude & airmass arrays
        altaz = coord.transform_to(AltAz(obstime=times,
                                         location=SITE_LOCATION))
        alts  = altaz.alt.degree
        zs    = 90 - alts
        zr    = np.radians(zs)
        Xs    = 1.0 / (
                   np.cos(zr)
                 + 0.50572*(6.07995 + zs)**(-1.6364)
                )

        # Plot
        norm = plt.Normalize(vmin=1, vmax=3)
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot_date(times.plot_date, alts, '-k', label='Altitude')
        sc = ax.scatter(
            times.plot_date,
            alts,
            c=Xs,
            cmap="coolwarm",
            norm=norm,
            s=25,
            alpha=0.8
        )

        ax.axhline(30, color='gray', ls='--', label='30° Elevation')
        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%H:%M", tz=tz_local)
        )
        ax.set_xlabel(f"Time ({LOCAL_TIMEZONE})")
        ax.set_ylabel("Altitude (°)")
        ax.set_ylim(0, 90)
        ax.grid(True)

        cbar = plt.colorbar(sc, ax=ax, pad=0.02, extend='max')
        cbar.set_label("Airmass (clamped at 3)")

        ax.legend(loc='lower right')
        plt.title(f"{name}: {RA} {DEC} — Altitude & Airmass on {date_str}")
        plt.tight_layout()
        plt.show()

    return X0

# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------
X_local = airmass_function(
    date_str="2025-10-01",
    hour=21,
    minute=0,
    RA="03:53:21",
    DEC="-00:00:20",
    name="area 95",
    input_timezone="Asia/Tehran",
    plot_night_curve=True
)
