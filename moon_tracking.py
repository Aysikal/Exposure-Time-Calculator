#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════#
#This code was written by Aysan Hemmatiortakand. Last updated 10/03/2025 
#you can contact me for any additional questions or information via Email 
#email address :aysanhemmatiortakand@gmail.com
#github = https://github.com/Aysikal
#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════#

from astropy.utils import iers
iers.conf.auto_max_age = None

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*predictive values that are more than.*",
    module="astropy.utils.iers"
)

from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_body
from astropy.time import Time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from astroplan import Observer
from astropy.io import fits
from zoneinfo import ZoneInfo
from datetime import datetime


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
def get_fli(date_str: str, hour: int, minute: int) -> float:
    """
    Fraction of the Moon illuminated (0–1) at the given local time
    in SITE_TIMEZONE. Also prints altitude & azimuth.

    Uses geocentric Sun–Moon elongation: k = (1 − cos ψ) / 2.
    """
    # Local → UTC
    tz_local = ZoneInfo(SITE_TIMEZONE)
    dt_local = datetime.strptime(f"{date_str} {hour:02d}:{minute:02d}", "%Y-%m-%d %H:%M").replace(tzinfo=tz_local)
    dt_utc   = dt_local.astimezone(ZoneInfo("UTC"))
    t_utc    = Time(dt_utc, scale="utc", location=SITE_LOCATION)

    # Geocentric Sun/Moon positions
    moon = get_body("moon", t_utc, SITE_LOCATION)
    sun  = get_body("sun",  t_utc, SITE_LOCATION)

    # Alt/Az for reporting
    moon_altaz = moon.transform_to(AltAz(obstime=t_utc, location=SITE_LOCATION))
    alt, az = moon_altaz.alt.deg, moon_altaz.az.deg

    # Elongation ψ (true sky separation)
    psi = sun.separation(moon).to(u.rad).value  # radians

    # Illuminated fraction
    fli = (1 - np.cos(psi)) / 2

    print(f"Time ({SITE_TIMEZONE}): {dt_local:%Y-%m-%d %H:%M}")
    print(f"Moon altitude: {alt:.2f}°")
    print(f"Moon azimuth:  {az:.2f}°")
    print(f"Elongation ψ:  {np.degrees(psi):.1f}°")
    print(f"Illuminated:   {fli*100:.1f}%")
    return fli


# Compute Moon fraction illuminated local time
#fli = get_fli("2025-10-01", 19, 0)


def overlay_visibility_with_moon(
    star_coords: dict,
    date: str,
    n_steps: int = 200,
    elevation_limit: u.Quantity = 30*u.deg,
    moon_alt_limit: u.Quantity = 20*u.deg,
    overlay: bool = True
):
    """
    Plot target altitude vs time and overlay the Moon’s altitude track
    for the night at SITE_LOCATION / SITE_OBSERVER.

    Parameters
    ----------
    star_coords : dict
        {"Name": (RA, DEC)}; RA/DEC as "HH:MM:SS"/"+DD:MM:SS" or floats (deg).
    date : str
        "YYYY-MM-DD" (interpreted in SITE_TIMEZONE for axis labeling).
    n_steps : int
        Resolution between sunset and sunrise.
    elevation_limit : Quantity
        Reference horizontal line (e.g. 30 deg).
    moon_alt_limit : Quantity
        Shade times when Moon altitude exceeds this threshold.
    overlay : bool
        True = all stars in one figure; False = one figure per star.
    """
    # Use SITE_OBSERVER and SITE_TIMEZONE defined globally
    tz_local = ZoneInfo(SITE_TIMEZONE)
    dt_mid_local = datetime.strptime(f"{date} 12:00", "%Y-%m-%d %H:%M").replace(tzinfo=tz_local)
    t_mid_utc = Time(dt_mid_local.astimezone(ZoneInfo("UTC")), scale="utc")

    sunset  = SITE_OBSERVER.sun_set_time(t_mid_utc, which="nearest")
    sunrise = SITE_OBSERVER.sun_rise_time(t_mid_utc, which="next")
    hours   = (sunrise - sunset).to(u.hour).value
    times   = sunset + np.linspace(0, hours, n_steps)*u.hour

    # Moon altitude curve
    moon_icrs  = get_body("moon", times, SITE_LOCATION)
    moon_altaz = moon_icrs.transform_to(AltAz(obstime=times, location=SITE_LOCATION))
    moon_alts  = moon_altaz.alt.degree
    mask_bright = moon_altaz.alt >= moon_alt_limit

    if overlay:
        fig, ax = plt.subplots(figsize=(10,6))
        # Shade bright intervals
        ax.fill_between(
            times.plot_date, 0, 90,
            where=mask_bright,
            color="gold", alpha=0.15, step="mid",
            label=f"Moon > {moon_alt_limit.to_value(u.deg):.0f}°"
        )

    # Plot each target
    for name, (ra, dec) in star_coords.items():
        unit = (u.hourangle, u.deg) if isinstance(ra, str) else (u.deg, u.deg)
        coord = SkyCoord(ra=ra, dec=dec, unit=unit, frame="icrs")
        altaz = coord.transform_to(AltAz(obstime=times, location=SITE_LOCATION))
        alts = altaz.alt.degree

        if overlay:
            ax.plot_date(times.plot_date, alts, "-", label=name)
        else:
            fig, ax_i = plt.subplots(figsize=(10,4.5))
            ax_i.plot_date(times.plot_date, alts, "-", label=name)
            ax_i.plot_date(times.plot_date, moon_alts, "-", color="goldenrod", label="Moon")
            ax_i.axhline(elevation_limit.to_value(u.deg), color="gray", ls="--", label=f"{elevation_limit}")
            ax_i.set_ylim(0, 90); ax_i.grid(True)
            ax_i.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=SITE_TIMEZONE))
            ax_i.set_xlabel(f"Time ({SITE_TIMEZONE})"); ax_i.set_ylabel("Altitude (°)")
            ax_i.legend(loc="lower right"); ax_i.set_title(f"{name} on {date} at {SITE_NAME}")
            plt.tight_layout(); plt.show()

    if overlay:
        # Overlay Moon track
        ax.plot_date(times.plot_date, moon_alts, "-", color="goldenrod", label="Moon")
        ax.axhline(elevation_limit.to_value(u.deg), color="gray", ls="--", label=f"{elevation_limit}")
        ax.set_ylim(0, 90); ax.grid(True)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=SITE_TIMEZONE))
        ax.set_xlabel(f"Time ({SITE_TIMEZONE})"); ax.set_ylabel("Altitude (°)")
        ax.legend(loc="lower right")
        ax.set_title(f"Targets + Moon on {date} at {SITE_NAME}")
        plt.tight_layout(); plt.show()
stars = {
    "vega" :  ("18:36:56.3", "+38:47:01")
}
#overlay_visibility_with_moon(stars, date="2025-10-01", moon_alt_limit=25*u.deg, overlay=True)

def moon_separation(
    date_str: str,
    hour: int,
    minute: int,
    ra: str,
    dec: str
) -> float:
    """
    Compute angular separation (degrees) between the Moon and a target
    at a given local time in SITE_TIMEZONE, using topocentric AltAz.

    Parameters
    ----------
    date_str : str
        Date in 'YYYY-MM-DD' (local to SITE_TIMEZONE).
    hour : int
        Hour (0–23) in SITE_TIMEZONE.
    minute : int
        Minute (0–59) in SITE_TIMEZONE.
    ra : str or float
        Target RA, e.g. "23:13:38.8" or degrees as float.
    dec : str or float
        Target DEC, e.g. "+39:25:02.6" or degrees as float.

    Returns
    -------
    float
        Angular separation in degrees (topocentric).
    """
    # Local → UTC
    tz_local = ZoneInfo(SITE_TIMEZONE)
    dt_local = datetime.strptime(f"{date_str} {hour:02d}:{minute:02d}",
                                 "%Y-%m-%d %H:%M").replace(tzinfo=tz_local)
    dt_utc   = dt_local.astimezone(ZoneInfo("UTC"))
    t_utc    = Time(dt_utc, scale="utc", location=SITE_LOCATION)

    # Target coordinates (ICRS → AltAz)
    unit = (u.hourangle, u.deg) if isinstance(ra, str) else (u.deg, u.deg)
    target_icrs = SkyCoord(ra=ra, dec=dec, unit=unit, frame="icrs")

    # Build AltAz frame for this time and site
    altaz_frame = AltAz(obstime=t_utc, location=SITE_LOCATION)

    # Transform both to AltAz
    target_altaz = target_icrs.transform_to(altaz_frame)
    moon_altaz   = get_body("moon", t_utc, SITE_LOCATION).transform_to(altaz_frame)

    # Separation in AltAz (true on-sky angle)
    sep = target_altaz.separation(moon_altaz).degree

    print(f"Time ({SITE_TIMEZONE}): {dt_local:%Y-%m-%d %H:%M}")
    print(f"Target: RA={ra}, DEC={dec}")
    print(f"Moon separation (AltAz): {sep:.2f}°")

    return sep
#moon_separation("2025-10-01", 20, 0, "18:36:56.3", "+38:47:01")  # Vega


# ─── Helper: airmass from altitude ────────────────────────────────────────────
def airmass_from_alt(alt_deg: float) -> float:
    z_deg = 90.0 - alt_deg
    z_rad = np.radians(z_deg)
    return 1.0 / (
        np.cos(z_rad)
        + 0.50572 * (6.07995 + z_deg) ** -1.6364
    )

# ─── Assume get_fli() and moon_separation() are defined above ────────────────

# Define your filter‐specific extinction coefficients (k) and zero‐points (zp_mag)
FILTER_PARAMS = {
    'u': { 'zp_mag': 22.7, 'k': 0.50 },   # SDSS u′ AB zero point ~22.7; k ~0.5
    'g': { 'zp_mag': 22.5, 'k': 0.20 },   # SDSS g′ AB zero point ~22.5; k ~0.2
    'r': { 'zp_mag': 22.4, 'k': 0.10 },   # SDSS r′ AB zero point ~22.4; k ~0.1
    'i': { 'zp_mag': 22.0, 'k': 0.07 },   # SDSS i′ AB zero point ~22.0; k ~0.07
    'V': { 'zp_mag': 21.6, 'k': 0.20 }    # Original V‐band defaults
}

def lunar_sky_brightness_at_target(
    date: str,
    hour: int,
    minute: int,
    ra: str,
    dec: str,
    filter_name: str      = 'g',         # choose 'u','g','r','i' or 'V'
    plot_sky_brightness: bool = False,   # show 2D sky‐brightness map
    n_alt: int            = 90,          # altitude resolution
    n_az: int             = 180          # azimuth resolution
) -> float:
    """Compute lunar sky brightness at a target (in mag/arcsec²)
    for any SDSS or V‐band filter and optionally plot the all‐sky map."""
    # ─── Lookup filter parameters ─────────────────────────────────────────────
    try:
        params = FILTER_PARAMS[filter_name]
    except KeyError:
        raise ValueError(f"Unknown filter {filter_name!r}; must be one of {list(FILTER_PARAMS)}")
    zp_mag = params['zp_mag']
    k      = params['k']

    # ─── Phase & illuminated fraction ─────────────────────────────────────────
    fli     = get_fli(date, hour, minute)
    psi_rad = np.arccos(1 - 2*fli)
    psi_deg = np.degrees(psi_rad)
    phi     = 10**(-0.4 * (3.84 + 0.026*abs(psi_deg) + 4e-9*psi_deg**4)) #eq 9,  KRISCIUNAS and SCHAEFER 1991

    # ─── Angular separation ───────────────────────────────────────────────────
    sep_deg   = moon_separation(date, hour, minute, ra, dec)
    theta_rad = np.radians(sep_deg)

    # ─── AltAz frame at local time ────────────────────────────────────────────
    tz_local    = ZoneInfo(SITE_TIMEZONE)
    dt_local    = datetime.strptime(f"{date} {hour:02d}:{minute:02d}",
                                    "%Y-%m-%d %H:%M") \
                       .replace(tzinfo=tz_local)
    t_utc       = Time(dt_local.astimezone(ZoneInfo("UTC")),
                       scale="utc", location=SITE_LOCATION)
    altaz_frame = AltAz(obstime=t_utc, location=SITE_LOCATION)

    # ─── Altitudes for Vega & Moon ────────────────────────────────────────────
    vega_altaz = SkyCoord(ra=ra, dec=dec,
                         unit=(u.hourangle, u.deg), frame="icrs") \
                    .transform_to(altaz_frame)
    alt_vega   = vega_altaz.alt.degree

    moon_altaz = get_body("moon", t_utc, SITE_LOCATION) \
                    .transform_to(altaz_frame)
    alt_moon = moon_altaz.alt.degree
    az_moon  = moon_altaz.az.degree

    # ─── Airmass & extinction ─────────────────────────────────────────────────
    X_vega    = airmass_from_alt(alt_vega)
    X_moon    = airmass_from_alt(alt_moon)
    ext_term  = (1 - 10**(-0.4 * k * X_moon)) \
                * 10**(-0.4 * k * X_vega)
    C_R = 1

    # ─── Scattering at target ─────────────────────────────────────────────────
    f_theta = C_R *(1.06 + np.cos(theta_rad)**2) #eq 19 , KRISCIUNAS and SCHAEFER 1991

    # ─── Point‐source brightness ───────────────────────────────────────────────
    B_moon = -2.5 * np.log10(phi * ext_term * f_theta) + zp_mag
    print(f"[{filter_name}′] Sky brightness at target: {B_moon:.2f} mag/arcsec²")

    # ─── Optional: all‐sky brightness map ─────────────────────────────────────
    if plot_sky_brightness:
        # ─ build alt/az grid ────────────────────────────────────────
        alts     = np.linspace(0, 90, n_alt)   # deg
        azs      = np.linspace(0, 360, n_az)   # deg
        alt_grid, az_grid = np.meshgrid(alts, azs, indexing='ij')

        # ─ separation angle per cell ─────────────────────────────────
        alt1    = np.radians(alt_grid)
        alt2    = np.radians(alt_moon)
        daz     = np.radians(az_grid - az_moon)
        cos_sep = (np.sin(alt1)*np.sin(alt2)
                   + np.cos(alt1)*np.cos(alt2)*np.cos(daz))
        sep_rad = np.arccos(np.clip(cos_sep, -1, 1))

        # ─ f(θ) scattering map ───────────────────────────────────────
        f_map = 1.06 + np.cos(sep_rad)**2

        # ─ full‐sky brightness map ───────────────────────────────────
        X_target = airmass_from_alt(alt_grid)
        ext_map  = (1 - 10**(-0.4 * k * X_moon)) \
                   * 10**(-0.4 * k * X_target)
        B_map    = -2.5 * np.log10(phi * ext_map * f_map) + zp_mag

        # ─ polar coords ──────────────────────────────────────────────
        theta    = np.radians(az_grid)
        r        = np.radians(90 - alt_grid)
        r_moon   = np.radians(90 - alt_moon)
        θ_moon   = np.radians(az_moon)

        # ─ set up figure ─────────────────────────────────────────────
        fig, (ax1, ax2) = plt.subplots(
            1, 2,
            subplot_kw={'projection':'polar'},
            figsize=(12, 6)
        )

        # ─ left: scattering f(θ) ────────────────────────────────────
        pcm1 = ax1.pcolormesh(
            theta, r, f_map,
            cmap='Blues_r',
            vmin=f_map.min(), vmax=f_map.max(),
            shading='auto'
        )
        ax1.scatter(θ_moon, r_moon,
                    color='white', edgecolor='black',
                    s=80, marker='o', label='Moon')  # changed to white "o"
        ax1.scatter(vega_altaz.az.radian, np.radians(90 - vega_altaz.alt.degree),
                    color='yellow', edgecolor='black',
                    s=80, marker='*', label='Target')  # added target marker
        ax1.set_title("Scattering f(θ)")
        ax1.set_theta_zero_location('N')
        ax1.set_theta_direction(-1)
        ax1.set_rmax(np.pi/2)
        cbar1 = fig.colorbar(pcm1, ax=ax1, pad=0.05)
        cbar1.set_label("f(θ)")

        # ─ right: full‐sky brightness ───────────────────────────────
        pcm2 = ax2.pcolormesh(
            theta, r, B_map,
            cmap='Blues',
            vmin= 30.5, vmax=31.5,
            shading='auto'
        )
        ax2.scatter(θ_moon, r_moon,
                    color='white', edgecolor='black',
                    s=80, marker='o', label='Moon')  # changed to white "o"
        ax2.scatter(vega_altaz.az.radian, np.radians(90 - vega_altaz.alt.degree),
                    color='yellow', edgecolor='black',
                    s=80, marker='*', label='Target')  # added target marker
        ax2.set_title(f"{filter_name}-band Sky Brightness (mag/arcsec²)")
        ax2.set_theta_zero_location('N')
        ax2.set_theta_direction(-1)
        ax2.set_rmax(np.pi/2)
        cbar2 = fig.colorbar(pcm2, ax=ax2, pad=0.05)
        cbar2.set_label("mag/arcsec²")

        ax1.legend(loc='upper right')
        ax2.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

    return B_moon

# Example usage:
vega_ra  = "18:36:56.3"
vega_dec = "+38:47:01"

# get g‐band brightness & plot
lunar_sky_brightness_at_target(
    date                = "2025-10-01",
    hour                = 20,
    minute              = 0,
    ra                  = vega_ra,
    dec                 = vega_dec,
    filter_name         = 'g',
    plot_sky_brightness = True
)
