import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import astropy.units as u
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta
from astropy.utils import iers
iers.conf.auto_download = False
iers.conf.auto_max_age = None

# --- Location: Tehran ---
tehran = EarthLocation(lat=35.6892*u.deg, lon=51.3890*u.deg, height=1200*u.m)

# --- Time range: 19:00 to 05:00 local ---
tz = ZoneInfo("Asia/Tehran")
start_local = datetime(2025, 10, 1, 19, 0, tzinfo=tz)
end_local   = datetime(2025, 10, 2, 5, 0, tzinfo=tz)

# Generate times every 10 minutes
times_local = [start_local + timedelta(minutes=10*i)
               for i in range(int((end_local-start_local).total_seconds()/600)+1)]
times_utc = [t.astimezone(ZoneInfo("UTC")) for t in times_local]
times = Time(times_utc)

# --- Vega coordinates ---
vega = SkyCoord.from_name("Vega")

# --- Transform to AltAz ---
altaz_frame = AltAz(obstime=times, location=tehran)
vega_altaz = vega.transform_to(altaz_frame)

# --- Polar coordinates ---
az = np.radians(vega_altaz.az.deg)
alt = vega_altaz.alt.deg
r = 90 - alt   # radius = 90° - altitude

# --- Set up figure ---
fig = plt.figure(figsize=(7,7))
ax = plt.subplot(111, polar=True)
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_rlim(90, 0)
ax.set_title("Vega Path from Tehran (19:00–05:00)", fontsize=14)

# Static path (faint background)
ax.plot(az, r, color="cyan", lw=1, alpha=0.3)

# Animated elements
(line,) = ax.plot([], [], color="cyan", lw=2, label="Vega path")
(point,) = ax.plot([], [], "ro", label="Current position")
time_text = ax.text(0.05, 0.05, "", transform=ax.transAxes)

ax.legend(loc="upper left")

# --- Animation function ---
def init():
    line.set_data([], [])
    point.set_data([], [])
    time_text.set_text("")
    return line, point, time_text

def update(frame):
    # up to current frame
    line.set_data(az[:frame], r[:frame])
    point.set_data(az[frame], r[frame])
    time_text.set_text(times_local[frame].strftime("%Y-%m-%d %H:%M"))
    return line, point, time_text

ani = FuncAnimation(fig, update, frames=len(times), init_func=init,
                    interval=200, blit=True, repeat=True)

plt.show()
# %% 

from skyfield.api import load, Star, Topos
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
from zoneinfo import ZoneInfo

# --- Setup ---
ts = load.timescale()
planets = load('de421.bsp')
earth = planets['earth']

# Vega coordinates (J2000)
vega = Star(ra_hours=(18, 36, 56.33635), dec_degrees=(38, 47, 1.2802))

# Observer in Tehran
observer = earth + Topos('35.6892 N', '51.3890 E')

# Time range: 19:00 to 05:00 local (Asia/Tehran = UTC+3:30)
tz = ZoneInfo("Asia/Tehran")
start_local = datetime(2025, 10, 1, 19, 0, tzinfo=tz)
end_local   = datetime(2025, 10, 2, 5, 0, tzinfo=tz)

# Generate times every 10 minutes
times_local = [start_local + timedelta(minutes=10*i)
               for i in range(int((end_local-start_local).total_seconds()/600)+1)]
times_utc = [t.astimezone(ZoneInfo("UTC")) for t in times_local]
times = [ts.from_datetime(t) for t in times_utc]

# Compute alt/az
records = []
for t, t_local in zip(times, times_local):
    alt, az, _ = observer.at(t).observe(vega).apparent().altaz()
    records.append({
        "Azimuth (°)": az.degrees,
        "Altitude (°)": alt.degrees,
        "Time": t_local.strftime("%H:%M")
    })

df = pd.DataFrame(records)

# --- Interactive animated plot ---
fig = px.scatter(df, x="Azimuth (°)", y="Altitude (°)",
                 animation_frame="Time", animation_group="Time",
                 range_x=[0,360], range_y=[0,90],
                 title="Vega Path from Tehran (19:00–05:00 local)",
                 labels={"Azimuth (°)":"Azimuth (°)", "Altitude (°)":"Altitude (°)"})

fig.update_traces(mode="markers+lines")
fig.show()
# %%
