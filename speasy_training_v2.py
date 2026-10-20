import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import speasy as spz
from scipy.ndimage import uniform_filter1d
from scipy.signal import get_window
from scipy.stats import chi2
from matplotlib.colors import LogNorm


deb = "2014-1-31 09:19"
fin = "2014-1-31 09:29"
mes_mag_test = spz.get_data('amda/mes_b_orbmso', deb, fin)

mes_xyz_test = spz.get_data('amda/mes_xyz_orbmso', deb, fin)


# Ions dont on peut tracer la fréquence cyclotron : (nombre de masse, charge)
IONS = {"H+": (1, 1), "He2+": (4, 2), "O+": (16, 1), "Na+": (23, 1)}
COULEURS_IONS = {"H+": "dimgray", "He2+": "goldenrod", "O+": "steelblue", "Na+": "#eb6834"}

def print_mes_mag(mes_mag):
    '''
    pour avoir de la visibilité sur les données extractables avec speazy :
    '''
    print("type =", type(mes_mag), "\ncolonnes=", mes_mag.columns, "\nunit=", mes_mag.unit)
    print("time dtype =", mes_mag.time.dtype)
    print("values dtype =", mes_mag.values.dtype)
    print("values shape =", mes_mag.values.shape)
    print("meta =", mes_mag.meta)


def mag(date_debut, date_fin):
    '''
    En entrée :
     - une date de début au format string "YYYY-MM-DD HH:mm"
     - une date de fin au format string   "YYYY-MM-DD HH:mm"

    Affiche l'évolution de Bx, By et Bz entre les dates de début et de fin, ainsi que |B|
    ''' 
    #Récupération des données
    mes_mag = spz.get_data('amda/mes_b_orbmso', date_debut, date_fin)
    if mes_mag is None:
        raise ValueError(f"Aucune donnée AMDA recuperee entre {date_debut} et {date_fin}")
 
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


def cyclo(nb_masse, B_moy, charge=1):
    """
    Calcul de la fréquence cyclotron d'un ion dans un champ magnétique
    Paramètres :
      - nb_masse : nombre de masse de l'ion (ex: 1 pour H+, 4 pour He++, 23 pour Na+)
                   attention : pas le numéro atomique (11 pour Na), qui doublerait f_cyclo
      - B_moy : champ magnétique moyen en nT (nombre ou tableau numpy)
      - charge : charge de l'ion (par défaut 1)
    """
    # Constantes
    e = 1.602176634e-19  # Charge élémentaire en C
    m_p = 1.67262192369e-27  # Masse du proton en kg

    # Masse de l'ion (approximation m = A m_p, erreur < 1 %)
    m_ion = nb_masse * m_p

    # Conversion du champ magnétique de nT à T
    B_moy_UI = B_moy * 1e-9

    # Calcul de la fréquence cyclotron en Hz
    f_cyclo = (charge * e * B_moy_UI) / (2 * np.pi * m_ion)

    return f_cyclo



def psd_mag(date_debut, date_fin, ions=("Na+", "H+"), fenetre=None, zoom_lin=0.25,
            donnees=None, trace=True):
    """
    En entrée :
      - date_debut : "YYYY-MM-DD HH:mm"
      - date_fin   : "YYYY-MM-DD HH:mm"
      - ions       : ions dont on trace la fréquence cyclotron (clés de IONS)
      - fenetre    : None (rectangulaire, comme Gershman et al. 2015) ou "hann" (pour mesurer des pentes)
      - zoom_lin   : borne du panneau en fréquence linéaire (None : pas de zoom)
      - donnees    : (t, B) automatiquement chargés à partir de l'extraction speasy
      - trace      : False pour seulement renvoyer les résultats (utilisé par spectro_mag)

    Affiche :
      - PSD parallèle au champ moyen
      - PSD perpendiculaire au champ moyen
      - fréquences cyclotron des ions, calculées avec <|B|> (bande : <|B|> ± écart-type)

    Renvoie un dict : f, PSD_perp, PSD_para, pola, deg_pola, fs, B_abs_moy, B_abs_std

    Méthode inspirée de Boardsen et al. (décrite dans Gershman et al. 2015, §2) :
      - detrending polynomial d'ordre 2
      - repère local (n1,n2,n3)
      - FFT directe
      - lissage sur 3 bins fréquentiels
    """

    # ==================================================
    # Chargement des données
    # ==================================================

    if donnees is None:
        mes_mag = spz.get_data('amda/mes_b_orbmso', date_debut, date_fin)
        if mes_mag is None:
            raise ValueError(f"Aucune donnée AMDA recuperee entre {date_debut} et {date_fin}")
        donnees = (mes_mag.time, mes_mag.values)

    t, B = donnees

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
            "Le champ moyen est presque parallèle à YMSO. Revoir le choix des axes locaux"
        )

    n1 /= np.linalg.norm(n1)

    n2 = np.cross(n3, n1)
    n2 /= np.linalg.norm(n2)

    # |B| moyen (et non |<B>|) pour les fréquences cyclotron, écart-type comme incertitude
    B_abs = np.linalg.norm(B, axis=1)
    B_abs_moy, B_abs_std = B_abs.mean(), B_abs.std()

    # ==================================================
    # Projection dans le repère local nouvellement établi
    # (produit matriciel pour se placer dans le repère local)
    # ==================================================

    B1 = B_det @ n1
    B2 = B_det @ n2
    B3 = B_det @ n3

    # ==================================================
    # FFT directe
    # ==================================================

    fs = 20.0  # Hz

    N = len(B1) # nombre d'échantillons

    # Fenêtre d'apodisation (rectangulaire par défaut : np.sum(w**2) = N)
    w = np.ones(N) if fenetre is None else get_window(fenetre, N)

    F1 = np.fft.rfft(B1 * w)
    F2 = np.fft.rfft(B2 * w)
    F3 = np.fft.rfft(B3 * w)

    f = np.fft.rfftfreq(N, d=1/fs)

    PSD1 = (2.0 / (fs * np.sum(w**2))) * np.abs(F1)**2
    PSD2 = (2.0 / (fs * np.sum(w**2))) * np.abs(F2)**2
    PSD3 = (2.0 / (fs * np.sum(w**2))) * np.abs(F3)**2

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

    # (bin 0 exclu : il vaut ~0 après detrending et tirerait le premier point vers le bas)
    PSD_perp[1:] = uniform_filter1d(
        PSD_perp[1:],
        size=3
    )

    PSD_para[1:] = uniform_filter1d(
        PSD_para[1:],
        size=3
    )

    # ==================================================
    # Polarisation dans le plan perpendiculaire
    # ==================================================
    # pola : +1 = droite (sens de giration des électrons autour de B), -1 = gauche (sens des ions)
    # deg_pola : ~1 pour une onde cohérente, plus faible pour du bruit

    S12 = (2.0 / (fs * np.sum(w**2))) * F1 * np.conj(F2)
    if N % 2 == 0:
        S12[-1] /= 2

    def lisse(x):
        return np.r_[np.nan, uniform_filter1d(x[1:], size=3)]

    P1_l, P2_l, C12_l, Q12_l = lisse(PSD1), lisse(PSD2), lisse(S12.real), lisse(S12.imag)
    pola = 2 * Q12_l / (P1_l + P2_l)
    deg_pola = np.sqrt(np.clip(1 - 4 * (P1_l * P2_l - C12_l**2 - Q12_l**2) / (P1_l + P2_l)**2, 0, 1))

    resultats = dict(f=f, PSD_perp=PSD_perp, PSD_para=PSD_para, pola=pola, deg_pola=deg_pola,
                     fs=fs, B_abs_moy=B_abs_moy, B_abs_std=B_abs_std)
    if not trace:
        return resultats

    # ==================================================
    # Intervalle de confiance à 95 %
    # ==================================================
    # Lissage sur 3 bins : nu = 6 degrés de liberté en rectangulaire,
    # moins avec Hann (bins voisins corrélés)

    W2 = np.abs(np.fft.fft(w**2))
    nu = 6 / (1 + 2 * ((2/3) * (W2[1] / W2[0])**2 + (1/3) * (W2[2] / W2[0])**2))
    ic_bas, ic_haut = nu / chi2.ppf(0.975, nu), nu / chi2.ppf(0.025, nu)

    def raies_cyclotron(f_min, f_max):
        """Fréquences cyclotron des ions, avec la bande <|B|> ± écart-type."""
        for ion in ions:
            A, q = IONS[ion]
            fc = cyclo(A, B_abs_moy, q)
            if f_min < fc < f_max:
                plt.axvspan(cyclo(A, B_abs_moy - B_abs_std, q), cyclo(A, B_abs_moy + B_abs_std, q),
                            color=COULEURS_IONS[ion], alpha=0.15, lw=0)
                plt.axvline(fc, color=COULEURS_IONS[ion], ls="--", lw=1.3)
                plt.text(fc, 0.98, f"{ion} : {fc:.3g} Hz ", transform=plt.gca().get_xaxis_transform(),
                         rotation=90, ha="right", va="top", fontsize=9)

    # ==================================================
    # Affichage
    # ==================================================

    plt.figure(figsize=(14, 6) if zoom_lin else (8, 6))
    if zoom_lin:
        plt.subplot(1, 2, 1)

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

    plt.xlim(f[1], f[-1])
    plt.axvspan(f[1], 2 * f[1], color="0.85", lw=0)  # bins atténués par le detrending
    raies_cyclotron(f[1], f[-1])

    # Barre d'intervalle de confiance en haut à droite
    y_bas, y_haut = plt.ylim()
    x_ic, y_ic = f[-1] / 2, y_haut / (y_haut / y_bas)**0.3
    plt.errorbar(x_ic, y_ic, yerr=[[y_ic * (1 - ic_bas)], [y_ic * (ic_haut - 1)]],
                 fmt="o", ms=3, color="k", capsize=3)
    plt.text(x_ic, y_ic * ic_haut * 1.2, "IC 95 %", ha="center", fontsize=8)
    plt.ylim(y_bas, y_haut)

    plt.xlabel("Frequency [Hz]")
    plt.ylabel(r"PSD [nT$^2$/Hz]")

    plt.grid(
        which="both",
        alpha=0.3
    )

    plt.legend()

    # Zoom en fréquence linéaire sur la bande KH / Na+ (format de Gershman et al. 2015, Fig. 6)
    if zoom_lin:
        plt.subplot(1, 2, 2)
        m = (f > 0) & (f <= zoom_lin)
        plt.semilogy(f[m], PSD_perp[m], color="purple", lw=1.5, marker="o", ms=2.5)
        plt.semilogy(f[m], PSD_para[m], color="mediumseagreen", lw=1.5, marker="o", ms=2.5)
        plt.xlim(0, zoom_lin)
        raies_cyclotron(0, zoom_lin)
        plt.xlabel("Frequency [Hz]")
        plt.title(rf"zoom, $\Delta f$ = {1e3 * f[1]:.1f} mHz")
        plt.grid(alpha=0.3)

    plt.suptitle(f"{str(t[0])[:19].replace('T', ' ')} → {str(t[-1])[11:19]} UTC | "
                 rf"$\langle|B|\rangle$ = {B_abs_moy:.1f} ± {B_abs_std:.1f} nT | fs = {fs:.2f} Hz")
    plt.tight_layout()
    plt.show()

    return resultats



def spectro_mag(date_debut, date_fin, duree_fen=300, pas=30, fmax=0.1, ion="Na+", seuil_pola=0.9):
    """
    En entrée :
      - date_debut, date_fin : "YYYY-MM-DD HH:mm"
      - duree_fen  : durée des fenêtres glissantes en s (5 min chez Gershman et al. 2015)
      - pas        : décalage entre deux fenêtres en s
      - fmax       : fréquence maximale affichée en Hz
      - ion        : ion dont on trace la fréquence cyclotron (|B| lissé sur 2.5 min)
      - seuil_pola : polarisation grisée là où le degré de polarisation est plus faible
                     (avec le lissage sur 3 bins, du bruit pur dépasse 0.7 dans ~40 % des cas)

    Affiche (cf. Gershman et al. 2015, Fig. 4) :
      - Bx, By, Bz et |B|
      - P_perp(t, f), chaque fenêtre étant traitée par psd_mag
      - polarisation : +1 droite (vortex KH), -1 gauche (ondes cyclotron ioniques)
    """
    mes_mag = spz.get_data('amda/mes_b_orbmso', date_debut, date_fin)
    if mes_mag is None:
        raise ValueError(f"Aucune donnée AMDA recuperee entre {date_debut} et {date_fin}")

    t, B = mes_mag.time, mes_mag.values
    ts = (t - t[0]) / np.timedelta64(1, "s")
    fs = 20.0  # Hz

    # Grille de fréquences commune : multiples de 1/duree_fen jusqu'à fmax
    f_ref = np.arange(1, int(fmax * duree_fen) + 1) / duree_fen
    debuts = np.arange(0, ts[-1] - duree_fen, pas) #??????
    P_perp, pola, deg_pola = [], [], []

    for d in debuts:
        sel = (ts >= d) & (ts < d + duree_fen)
        if np.sum(np.isfinite(B[sel, 0])) < 0.9 * duree_fen * fs:  # fenêtre trop trouée
            r = dict(f=f_ref, PSD_perp=np.full(f_ref.size, np.nan), pola=np.full(f_ref.size, np.nan),
                     deg_pola=np.full(f_ref.size, np.nan))
        else:
            r = psd_mag(None, None, donnees=(t[sel], B[sel]), trace=False)
        P_perp.append(np.interp(f_ref, r["f"][1:], r["PSD_perp"][1:]))
        pola.append(np.interp(f_ref, r["f"][1:], r["pola"][1:]))
        deg_pola.append(np.interp(f_ref, r["f"][1:], r["deg_pola"][1:]))

    P_perp, pola, deg_pola = np.array(P_perp), np.array(pola), np.array(deg_pola)
    centres = mdates.date2num(t[0] + ((debuts + duree_fen / 2) * 1e9).astype("timedelta64[ns]"))

    # Fréquence cyclotron le long de l'orbite, |B| moyenné sur 2.5 min
    B_abs = np.linalg.norm(B, axis=1)
    ok = np.isfinite(B_abs)
    B_lisse = uniform_filter1d(np.interp(ts, ts[ok], B_abs[ok]), size=int(150 * fs))
    A, q = IONS[ion]
    f_ion = cyclo(A, B_lisse, q)

    # Plot

    fig, axs = plt.subplots(3, 2, figsize=(11, 9), sharex="col", constrained_layout=True,
                            gridspec_kw=dict(width_ratios=[1, 0.02]))
    axs[0, 1].axis("off")
    x = mdates.date2num(t)

    for k, lab in enumerate([r"$B_x$", r"$B_y$", r"$B_z$"]):
        axs[0, 0].plot(x, B[:, k], lw=0.6, label=lab)
    axs[0, 0].plot(x, B_abs, color="k", lw=0.8, label=r"$|B|$")
    axs[0, 0].axhline(0, color="k", ls=":", lw=0.8)
    axs[0, 0].set_ylabel("B MSO (nT)")
    leg = axs[0, 0].legend(ncol=4, loc="upper right", fontsize=9)
    for ligne in leg.get_lines():
        ligne.set_linewidth(2)

    pcm = axs[1, 0].pcolormesh(centres, f_ref, P_perp.T, shading="nearest", cmap="viridis",
                               norm=LogNorm(*np.nanpercentile(P_perp, [2, 99.5])))
    fig.colorbar(pcm, cax=axs[1, 1], label=r"$P_\perp$ [nT$^2$/Hz]")

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("0.85")  # gris là où la polarisation n'a pas de sens
    pola_aff = np.ma.masked_where(~(deg_pola >= seuil_pola), pola)
    pcm = axs[2, 0].pcolormesh(centres, f_ref, pola_aff.T, shading="nearest", cmap=cmap, vmin=-1, vmax=1)
    fig.colorbar(pcm, cax=axs[2, 1], label="polarisation\n(+1 droite, -1 gauche)")

    for ax, couleur in ((axs[1, 0], "w"), (axs[2, 0], "k")):
        ax.plot(x, f_ion, color=couleur, lw=1.5, label=f"$f_c$ {ion}")
        ax.set_ylim(0, fmax)
        ax.set_ylabel("Frequency [Hz]")
        ax.legend(loc="upper right", fontsize=9, facecolor="0.7")

    axs[2, 0].set_xlim(x[0], x[-1])
    axs[2, 0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axs[2, 0].set_xlabel("UTC")
    plt.show()


'''Fonctions à lancer'''
#mag(deb, fin)
#psd_mag(deb, fin)
#spectro_mag("2014-1-31 09:09", "2014-1-31 09:39")
#print_mes_mag(mes_mag_test)
#print("serie temporelle :\n", mes_xyz_test.time, "\n")
#print("série position :\n", mes_xyz_test.values)
#print_mes_mag(mes_xyz_test)
print(mes_xyz_test.units)