import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import speasy as spz
from scipy.ndimage import uniform_filter1d


deb = "2014-1-31 09:19"
fin = "2014-1-31 09:29"


'''
En entrée :
 - une date de début au format string "YYYY-MM-DD HH:mm"
 - une date de fin au format string   "YYYY-MM-DD HH:mm"

Affiche l'évolution de Bx, By et Bz entre les dates de début et de fin, ainsi que |B|
'''
def mag(date_debut, date_fin):
    
    #Récupération des données
    
    mes_mag = spz.get_data('amda/mes_b_orbmso', date_debut, date_fin)

    t = mes_mag.time
    B = mes_mag.values
    norm = np.linalg.norm(B, axis=1)

    labels = [r"$B_x$ (nT)", r"$B_y$ (nT)", r"$B_z$ (nT)", r"$|B|$ (nT)"]
    series = [B[:, 0], B[:, 1], B[:, 2], norm]


    # Plot
    
    fig, axes = plt.subplots(4, 1, figsize=(9, 8), sharex=True,gridspec_kw={"hspace": 0.08})

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



def psd_mag(date_debut, date_fin):
    """
    En entrée :
      - date_debut : "YYYY-MM-DD HH:mm"
      - date_fin   : "YYYY-MM-DD HH:mm"

    Affiche :
      - PSD parallèle au champ moyen
      - PSD perpendiculaire au champ moyen

    Méthode inspirée de Boardsen et al. :
      - detrending polynomial d'ordre 2
      - repère local (n1,n2,n3)
      - FFT directe
      - lissage sur 3 bins fréquentiels
    """

    # ==================================================
    # Chargement des données
    # ==================================================

    mes_mag = spz.get_data('amda/mes_b_orbmso', date_debut, date_fin)

    B = mes_mag.values

    # ==================================================
    # Detrending polynomial ordre 2
    # ==================================================

    time_index = np.arange(len(B))

    B_det = np.zeros_like(B)

    for k in range(3):
        p = np.polyfit(time_index, B[:, k], deg=2)
        trend = np.polyval(p, time_index)
        B_det[:, k] = B[:, k] - trend

    # ==================================================
    # Champ moyen
    # ==================================================

    B_mean = np.mean(B, axis=0)

    n3 = B_mean / np.linalg.norm(B_mean)

    # Axe YMSO
    YMSO = np.array([0.0, 1.0, 0.0])

    n1 = np.cross(n3, YMSO)

    if np.linalg.norm(n1) < 1e-12:
        raise ValueError(
            "Le champ moyen est presque parallèle à YMSO."
        )

    n1 /= np.linalg.norm(n1)

    n2 = np.cross(n3, n1)
    n2 /= np.linalg.norm(n2)

    # ==================================================
    # Projection dans le repère local
    # ==================================================

    B1 = B_det @ n1
    B2 = B_det @ n2
    B3 = B_det @ n3

    # ==================================================
    # FFT directe
    # ==================================================

    fs = 20.0  # Hz

    N = len(B1)

    F1 = np.fft.rfft(B1)
    F2 = np.fft.rfft(B2)
    F3 = np.fft.rfft(B3)

    f = np.fft.rfftfreq(N, d=1/fs)

    PSD1 = (2.0 / (fs * N)) * np.abs(F1)**2
    PSD2 = (2.0 / (fs * N)) * np.abs(F2)**2
    PSD3 = (2.0 / (fs * N)) * np.abs(F3)**2

    # Correction composante continue
    PSD1[0] /= 2
    PSD2[0] /= 2
    PSD3[0] /= 2

    # Correction fréquence de Nyquist
    if N % 2 == 0:
        PSD1[-1] /= 2
        PSD2[-1] /= 2
        PSD3[-1] /= 2

    # ==================================================
    # PSD parallèle et perpendiculaire
    # ==================================================

    PSD_perp = 0.5 * (PSD1 + PSD2)
    PSD_para = PSD3

    # ==================================================
    # Lissage (3 bins)
    # ==================================================

    PSD_perp = uniform_filter1d(
        PSD_perp,
        size=3
    )

    PSD_para = uniform_filter1d(
        PSD_para,
        size=3
    )

    # ==================================================
    # Affichage
    # ==================================================

    plt.figure(figsize=(8, 6))

    plt.loglog(
        f[1:],
        PSD_perp[1:],
        color="purple",
        lw=1.5,
        label=r"$P_\perp=\frac{1}{2}(P_1+P_2)$"
    )

    plt.loglog(
        f[1:],
        PSD_para[1:],
        color="mediumseagreen",
        lw=1.5,
        label=r"$P_\parallel=P_3$"
    )

    plt.xlabel("Frequency [Hz]")
    plt.ylabel(r"PSD [nT$^2$/Hz]")

    plt.grid(
        which="both",
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()
    plt.show()





#mag(deb, fin)
psd_mag(deb, fin)