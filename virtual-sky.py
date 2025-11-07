import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from astropy.coordinates import SkyCoord, AltAz, EarthLocation, get_sun, get_body
from astropy.time import Time
import astropy.units as u

# ----------------------
# Site info
# ----------------------
SITE_NAME     = "Iran National Observatory"
SITE_LAT      = 33.674
SITE_LON      = 51.3188
SITE_ELEV     = 3600

location = EarthLocation(lat=SITE_LAT*u.deg, lon=SITE_LON*u.deg, height=SITE_ELEV*u.m)

# ----------------------
# Manual date input
# ----------------------
date_input = input("Enter date and time (YYYY-MM-DD HH:MM:SS, local Iran time): ")
time_local = Time(date_input)
time_utc = time_local - 3.5*u.hour  # Iran time -> UTC

altaz_frame = AltAz(obstime=time_utc, location=location)

# ----------------------
# Sun and Moon
# ----------------------
sun_altaz = get_sun(time_utc).transform_to(altaz_frame)
moon_altaz = get_body("moon", time_utc).transform_to(altaz_frame)

# ----------------------
# Helper functions
# ----------------------
def altaz_to_xyz(alt, az):
    """Convert Alt/Az to 3D Cartesian coordinates on unit sphere."""
    alt_rad = np.radians(alt)
    az_rad = np.radians(az)
    x = np.cos(alt_rad) * np.cos(az_rad)
    y = np.cos(alt_rad) * np.sin(az_rad)
    z = np.sin(alt_rad)
    return x, y, z

def filter_upper(x, y, z):
    """Keep only points with z >= 0 (above horizon)."""
    mask = z >= 0
    return x[mask], y[mask], z[mask]

# ----------------------
# Prepare 3D figure
# ----------------------
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor("white")  # White background

# Remove 3D panes and grid
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(False)

# ----------------------
# Plot Sun
# ----------------------
x, y, z = altaz_to_xyz(sun_altaz.alt.deg, sun_altaz.az.deg)
x, y, z = filter_upper(np.array([x]), np.array([y]), np.array([z]))
ax.scatter(x, y, z, color='orange', s=200, label='Sun')

# ----------------------
# Plot Moon
# ----------------------
x, y, z = altaz_to_xyz(moon_altaz.alt.deg, moon_altaz.az.deg)
x, y, z = filter_upper(np.array([x]), np.array([y]), np.array([z]))
ax.scatter(x, y, z, color='black', s=100, label='Moon')

# ----------------------
# Plot RA/Dec grid and labels
# ----------------------
ra_lines = np.arange(0, 360, 30)
dec_lines = np.arange(-90, 91, 30)

# RA lines with better label placement
for ra in ra_lines:
    decs = np.linspace(-90, 90, 180)
    ras = np.full_like(decs, ra)
    coords = SkyCoord(ra=ras*u.deg, dec=decs*u.deg, frame='icrs')
    altaz = coords.transform_to(altaz_frame)
    x, y, z = altaz_to_xyz(altaz.alt.deg, altaz.az.deg)
    x, y, z = filter_upper(x, y, z)
    ax.plot(x, y, z, color='blue', alpha=0.7, linewidth=0.8)
    
    if len(x) > 0:
        # Pick a point ~1/4th along the line (not top)
        idx = len(x) // 4
        ax.text(x[idx], y[idx], z[idx]+0.02, f'RA={ra}°', color='blue', fontsize=8)

# Dec lines
for dec in dec_lines:
    ras = np.linspace(0, 360, 360)
    decs = np.full_like(ras, dec)
    coords = SkyCoord(ra=ras*u.deg, dec=decs*u.deg, frame='icrs')
    altaz = coords.transform_to(altaz_frame)
    x, y, z = altaz_to_xyz(altaz.alt.deg, altaz.az.deg)
    x, y, z = filter_upper(x, y, z)
    ax.plot(x, y, z, color='green', alpha=0.7, linewidth=0.8)
    if len(x) > 0:
        mid = len(x)//2
        ax.text(x[mid], y[mid], z[mid]+0.02, f'Dec={dec}°', color='green', fontsize=8)

# ----------------------
# Adjust 3D view
# ----------------------
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([0, 1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.title(f"3D Sky from {SITE_NAME} on {date_input}", color="black")

plt.show()
