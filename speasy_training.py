import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import speasy as spz

mes_mag = spz.get_data('amda/mes_b_orbmso', "2011-6-17 00:46", "2011-6-17 01:00")

t = mes_mag.time
B = mes_mag.values
norm = np.linalg.norm(B, axis=1)

labels = [r"$B_x$ (nT)", r"$B_y$ (nT)", r"$B_z$ (nT)", r"$|B|$ (nT)"]
series = [B[:, 0], B[:, 1], B[:, 2], norm]

fig, axes = plt.subplots(4, 1, figsize=(9, 8), sharex=True,
                         gridspec_kw={"hspace": 0.08})

for i, (ax, y, lab) in enumerate(zip(axes, series, labels)):
    ax.plot(t, y, color="k", lw=0.8)
    ax.set_ylabel(lab)
    ax.tick_params(direction="in", top=True, right=True)
    if i < 3:  # pointillés à zéro pour Bx, By, Bz uniquement
        ax.axhline(0, color="k", ls=":", lw=0.8)

# Axe temps au format h:mm:ss, un tick toutes les 2 minutes
def fmt(x, pos=None):
    d = mdates.num2date(x)
    return f"{d.hour}:{d.minute:02d}:{d.second:02d}"

axes[-1].xaxis.set_major_locator(mdates.MinuteLocator(interval=2))
axes[-1].xaxis.set_major_formatter(plt.FuncFormatter(fmt))
axes[-1].set_xlim(t[0], t[-1])
axes[-1].text(0, 0, "UTC", transform=axes[-1].transAxes,
              ha="right", va="top")

plt.show()