#!/usr/bin/env python3
"""
traitement.py : figures « tout en un » pour une liste d'évènements KH (MESSENGER, données AMDA via speasy).

Pour chaque évènement de events.py, produit UNE figure contenant :
  - Bx, By, Bz, |B| (repère MSO), l'intervalle de l'évènement étant surligné
  - la PSD parallèle / perpendiculaire (log-log + zoom linéaire) avec les fréquences cyclotron
    de H+, He2+, O+ et Na+
  - l'orbite du vaisseau en deux projections 2D (X-Y vue du nord, X-Z vue depuis l'aube) avec la
    magnétopause moyenne, le sens de parcours et le sens de rotation de Mercure (aube / crépuscule)
  - la position en fonction du temps : altitude, latitude, heure locale

Fenêtre d'analyse : l'intervalle de l'évènement est élargi symétriquement pour que la fenêtre
totale vaille FACTEUR_FENETRE fois sa durée (B, PSD et orbite utilisent cette même fenêtre).

Organisation de data/ (créé à côté de ce fichier) :
  data/cache/                   données brutes téléchargées (.npz), réécrites à chaque exécution (sauf --no-force)
  data/<groupe>/<evenement>.png figures
  data/<groupe>/<groupe>_recap.pdf  (option --pdf) toutes les figures du groupe en un PDF
  data/manuel/<evenement>.png   figure d'un évènement ponctuel (option --evenement)
  data/log.txt                  journal des évènements traités / ignorés / en erreur

Usage :
  python traitement.py                       tous les groupes
  python traitement.py --groupes Sundberg    un seul groupe
  python traitement.py --limite 3            3 premiers évènements de chaque groupe (test)
  python traitement.py --no-force            réutilise le cache data/cache/ au lieu de retélécharger depuis AMDA
                                             (par défaut, les données sont toujours réimportées)
  python traitement.py --pdf                 ajoute un PDF récapitulatif par groupe
  python traitement.py --evenement "2011-05-15 09:26" "2011-05-15 09:31"
                                             un seul évènement (début, fin), hors events.py,
                                             PNG dans data/manuel/ ; fonctionne avec --no-force
"""
import argparse
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # pas de fenêtre : on écrit directement dans des fichiers
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Wedge
from scipy.ndimage import uniform_filter1d
from scipy.stats import chi2

import events as evmod

# ======================================================================
# Configuration
# ======================================================================
RACINE = Path(__file__).resolve().parent
DATA = RACINE / "data"
CACHE = DATA / "cache"

# Groupes d'évènements (nom du sous-dossier -> liste d'Event). Ajouter ici de nouvelles listes.
GROUPES = {
    "Sundberg": evmod.events_Sundberg,
    "Gershman_A1": evmod.events_Gershman_A1,
    "Gershman_A2": evmod.events_Gershman_A2,
}

PRODUIT_B = "amda/mes_b_orbmso"
PRODUIT_XYZ = "amda/mes_xyz_orbmso"

# Plage de définition de mes_b_orbmso sur AMDA (donnée en orbite : les survols de 2008-2009 n'y sont pas)
PLAGE_B = (datetime(2011, 3, 23, 15, 36, 40), datetime(2015, 4, 30, 15, 8, 14))

FACTEUR_FENETRE = 2.0    # fenêtre totale = 2 x durée de l'évènement, répartie symétriquement
FS_DEFAULT = 20.0         # Hz, utilisé si la cadence ne peut pas être estimée
ZOOM_LIN = 0.25          # Hz, borne du panneau PSD en fréquence linéaire
MAX_TROU_S = 1.0         # trous de B comblés (moyenne des deux voisins) jusqu'à cette durée ; au-delà : PSD refusée
DPI = 150

# Ions dont on trace la fréquence cyclotron : (nombre de masse, charge)
IONS = {"H+": (1, 1), "He2+": (4, 2), "O+": (16, 1), "Na+": (23, 1)}
COULEURS_IONS = {"H+": "dimgray", "He2+": "goldenrod", "O+": "steelblue", "Na+": "#eb6834"}

# Magnétopause moyenne de Winslow et al. (2013)
R_SS, ALPHA, Z_DIP = 1.45, 0.5, 0.196
R_MERCURE_KM = 2439.7


class DonneesInsuffisantes(Exception):
    """Données absentes ou trop lacunaires pour l'analyse demandée."""


# ======================================================================
# Dates, noms
# ======================================================================
def parse(s):
    return datetime.strptime(s.strip(), "%Y-%m-%d %H:%M")


def fenetre_elargie(deb, fin):
    """Fenêtre de durée FACTEUR_FENETRE x (fin - deb), centrée sur l'évènement."""
    marge = (fin - deb) * ((FACTEUR_FENETRE - 1) / 2)
    return (deb - marge).replace(microsecond=0), (fin + marge).replace(microsecond=0)


def nom_evenement(deb, fin):
    fin_txt = f"{fin:%H%M}" if fin.date() == deb.date() else f"{fin:%Y%m%d_%H%M}"
    return f"{deb:%Y-%m-%d_%H%M}-{fin_txt}"


# ======================================================================
# Journal
# ======================================================================
def journal(msg):
    ligne = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(ligne)
    DATA.mkdir(exist_ok=True)
    with open(DATA / "log.txt", "a", encoding="utf-8") as fh:
        fh.write(ligne + "\n")


# ======================================================================
# Collecte (avec cache)
# ======================================================================
def telecharger(produit, d, f, force=False):
    """
    Renvoie (t, valeurs) pour un produit AMDA entre les datetimes d et f, ou None si aucune donnée.
    Le résultat est écrit dans data/cache/. Avec force=False et un fichier présent, il est relu sans toucher au réseau.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    nom = f"{produit.split('/')[-1]}_{d:%Y%m%dT%H%M%S}_{f:%Y%m%dT%H%M%S}.npz"
    chemin = CACHE / nom
    if chemin.exists() and not force:
        z = np.load(chemin)
        return z["t"], z["v"]

    import speasy as spz  # import tardif : inutile si tout est déjà en cache
    res = spz.get_data(produit, f"{d:%Y-%m-%dT%H:%M:%S}", f"{f:%Y-%m-%dT%H:%M:%S}")
    if res is None or len(res.time) == 0:
        return None
    t = np.asarray(res.time).astype("datetime64[ns]")
    v = np.asarray(res.values, dtype=float)
    np.savez_compressed(chemin, t=t, v=v)
    return t, v


# ======================================================================
# Calculs
# ======================================================================
def _trous(nan):
    """Débuts et fins (fin exclue) des séquences consécutives de True dans un masque booléen."""
    d = np.diff(np.r_[0, nan.astype(int), 0])
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def plus_long_trou(B):
    """Longueur (en échantillons) du plus long trou, une ligne étant manquante si une composante l'est."""
    deb, fin = _trous(~np.isfinite(B).all(axis=1))
    return int((fin - deb).max()) if len(deb) else 0


def combler_trous(B, max_trou):
    """
    Remplace chaque valeur manquante (NaN) par la moyenne de ses deux voisines valides, composante par composante.
    Pour une séquence de plusieurs NaN consécutifs, toute la séquence reçoit la moyenne des deux valeurs valides
    qui l'encadrent ; en bord de série (une seule voisine), on recopie cette voisine.
    Les séquences plus longues que max_trou échantillons ne sont pas touchées.
    Renvoie une copie comblée de B.
    """
    B = B.copy()
    for k in range(B.shape[1]):
        deb, fin = _trous(~np.isfinite(B[:, k]))
        for a, b in zip(deb, fin):
            if b - a > max_trou:
                continue
            voisines = [B[i, k] for i in (a - 1, b) if 0 <= i < len(B)]
            if not voisines:
                continue
            B[a:b, k] = np.mean(voisines)
    return B


def cyclo(nb_masse, B_nT, charge=1):
    """Fréquence cyclotron (Hz) d'un ion de nombre de masse nb_masse dans un champ B (nT)."""
    e = 1.602176634e-19
    m_p = 1.67262192369e-27
    return charge * e * (B_nT * 1e-9) / (2 * np.pi * nb_masse * m_p)


def calcul_psd(t, B):
    """
    PSD parallèle et perpendiculaire au champ moyen, méthode de Gershman et al. (2015, §2) :
    detrending polynomial d'ordre 2, repère local (n1, n2, n3 // <B>), FFT rectangulaire,
    lissage sur 3 bins. Renvoie un dict.
    """
    N = len(B)
    dt = np.median(np.diff(t).astype("timedelta64[ns]").astype(float)) * 1e-9 if N > 1 else 0
    fs = 1.0 / dt if dt > 0 else FS_DEFAULT
    if N < 256:
        raise DonneesInsuffisantes(f"PSD impossible : seulement {N} échantillons")
    if not np.isfinite(B).all():  # il reste des trous plus longs que MAX_TROU_S : la FFT exige une série continue
        trou = plus_long_trou(B)
        raise DonneesInsuffisantes(
            f"PSD impossible : trou de {trou} échantillons ({trou / fs:.1f} s) non comblé (> {MAX_TROU_S:g} s)")

    # Detrending polynomial d'ordre 2 (abscisse normalisée pour le conditionnement)
    x = np.linspace(-1, 1, N)
    B_det = np.column_stack([B[:, k] - np.polyval(np.polyfit(x, B[:, k], 2), x) for k in range(3)])

    # Repère local : n3 // <B>, n1 = n3 x Y_MSO, n2 = n3 x n1
    n3 = B.mean(axis=0)
    n3 /= np.linalg.norm(n3)
    n1 = np.cross(n3, [0.0, 1.0, 0.0])
    if np.linalg.norm(n1) < 1e-12:
        raise DonneesInsuffisantes("champ moyen quasi parallèle à Y_MSO : repère local indéfini")
    n1 /= np.linalg.norm(n1)
    n2 = np.cross(n3, n1)
    n2 /= np.linalg.norm(n2)

    f = np.fft.rfftfreq(N, d=1 / fs)
    P = []
    for n in (n1, n2, n3):
        p = (2.0 / (fs * N)) * np.abs(np.fft.rfft(B_det @ n)) ** 2
        p[0] /= 2                 # composante continue
        if N % 2 == 0:
            p[-1] /= 2            # fréquence de Nyquist
        P.append(p)
    perp = 0.5 * (P[0] + P[1])
    para = P[2]
    perp[1:] = uniform_filter1d(perp[1:], size=3)
    para[1:] = uniform_filter1d(para[1:], size=3)

    B_abs = np.linalg.norm(B, axis=1)
    return dict(f=f, perp=perp, para=para, fs=fs, N=N,
                B_moy=B_abs.mean(), B_std=B_abs.std())


def magnetopause():
    """Section méridienne de la magnétopause de Winslow : (x, rho) avec rho = distance à l'axe du dipôle."""
    th = np.radians(np.linspace(0, 140, 200))
    r = R_SS * (2 / (1 + np.cos(th))) ** ALPHA
    return r * np.cos(th), r * np.sin(th)


# ======================================================================
# Tracés
# ======================================================================
def tracer_b(axs, t, B, deb, fin, wd, wf):
    B_abs = np.linalg.norm(B, axis=1)
    series = [B[:, 0], B[:, 1], B[:, 2], B_abs]
    labels = [r"$B_x$ (nT)", r"$B_y$ (nT)", r"$B_z$ (nT)", r"$|B|$ (nT)"]
    for i, (ax, y, lab) in enumerate(zip(axs, series, labels)):
        ax.plot(t, y, color="k", lw=0.8)
        ax.axvspan(deb, fin, color="tab:orange", alpha=0.15, lw=0,
                   label="évènement" if i == 0 else None)
        ax.set_ylabel(lab)
        ax.tick_params(direction="in", top=True, right=True)
        if i < 3:
            ax.axhline(0, color="k", ls=":", lw=0.8)
    axs[0].set_xlim(wd, wf)
    axs[0].legend(loc="upper right", fontsize=8)
    for ax in axs[:-1]:
        ax.tick_params(labelbottom=False)
    axs[-1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    axs[-1].set_xlabel(f"UTC  ({wd:%Y-%m-%d})")


def raies_cyclotron(ax, r, f_min, f_max):
    """Fréquences cyclotron de tous les ions de IONS, avec la bande <|B|> ± écart-type."""
    for ion, (A, q) in IONS.items():
        fc = cyclo(A, r["B_moy"], q)
        if f_min < fc < f_max:
            ax.axvspan(cyclo(A, r["B_moy"] - r["B_std"], q), cyclo(A, r["B_moy"] + r["B_std"], q),
                       color=COULEURS_IONS[ion], alpha=0.15, lw=0)
            ax.axvline(fc, color=COULEURS_IONS[ion], ls="--", lw=1.2)
            ax.text(fc, 0.98, f"{ion} : {fc:.3g} Hz ", transform=ax.get_xaxis_transform(),
                    rotation=90, ha="right", va="top", fontsize=8)


def tracer_psd(ax_log, ax_lin, r):
    f, perp, para = r["f"], r["perp"], r["para"]

    ax_log.loglog(f[1:], perp[1:], color="purple", lw=1.3, label=r"$P_\perp$")
    ax_log.loglog(f[1:], para[1:], color="mediumseagreen", lw=1.3, label=r"$P_\parallel$")
    ax_log.set_xlim(f[1], f[-1])
    ax_log.axvspan(f[1], 2 * f[1], color="0.85", lw=0)  # bins atténués par le detrending
    raies_cyclotron(ax_log, r, f[1], f[-1])

    # Barre d'intervalle de confiance à 95 % (lissage sur 3 bins, fenêtre rectangulaire : 6 ddl)
    nu = 6
    ic_bas, ic_haut = nu / chi2.ppf(0.975, nu), nu / chi2.ppf(0.025, nu)
    y_bas, y_haut = ax_log.get_ylim()
    x_ic, y_ic = f[-1] / 2, y_haut / (y_haut / y_bas) ** 0.3
    ax_log.errorbar(x_ic, y_ic, yerr=[[y_ic * (1 - ic_bas)], [y_ic * (ic_haut - 1)]],
                    fmt="o", ms=3, color="k", capsize=3)
    ax_log.text(x_ic, y_ic * ic_haut * 1.2, "IC 95 %", ha="center", fontsize=8)
    ax_log.set_ylim(y_bas, y_haut)
    ax_log.set_xlabel("Fréquence [Hz]")
    ax_log.set_ylabel(r"PSD [nT$^2$/Hz]")
    ax_log.grid(which="both", alpha=0.3)
    ax_log.legend(loc="lower left", fontsize=8)
    ax_log.set_title(rf"$\langle|B|\rangle$ = {r['B_moy']:.1f} ± {r['B_std']:.1f} nT, "
                     rf"$f_s$ = {r['fs']:.1f} Hz", fontsize=9)

    m = (f > 0) & (f <= ZOOM_LIN)
    ax_lin.semilogy(f[m], perp[m], color="purple", lw=1.3, marker="o", ms=2.5)
    ax_lin.semilogy(f[m], para[m], color="mediumseagreen", lw=1.3, marker="o", ms=2.5)
    ax_lin.set_xlim(0, ZOOM_LIN)
    raies_cyclotron(ax_lin, r, 0, ZOOM_LIN)
    ax_lin.set_xlabel("Fréquence [Hz]")
    ax_lin.set_ylabel(r"PSD [nT$^2$/Hz]")
    ax_lin.set_title(rf"zoom, $\Delta f$ = {1e3 * f[1]:.1f} mHz", fontsize=9)
    ax_lin.grid(alpha=0.3)


def message_axe(ax, texte):
    ax.text(0.5, 0.5, texte, ha="center", va="center", transform=ax.transAxes, color="firebrick")
    ax.set_xticks([])
    ax.set_yticks([])


def fleche_parcours(ax, u, v):
    """Flèche au milieu de la trajectoire (donc au centre de l'évènement) dans le sens du temps."""
    n = len(u)
    if n < 2:
        return
    m = max(1, n // 10)
    a, b = max(0, n // 2 - m), min(n - 1, n // 2 + m)
    ax.annotate("", xy=(u[b], v[b]), xytext=(u[a], v[a]), zorder=6,
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.6, mutation_scale=15,
                                shrinkA=0, shrinkB=0))


def rotation_planete_xy(ax):
    """
    Vue X-Y (depuis le nord, Soleil à droite) : Mercure tourne dans le sens prograde, donc antihoraire.
    En haut (+Y) la surface va du jour vers la nuit : coucher du Soleil, crépuscule.
    En bas (-Y) elle va de la nuit vers le jour : lever du Soleil, aube.
    """
    style = dict(arrowstyle="-|>", color="k", lw=1.3, mutation_scale=11, shrinkA=0, shrinkB=0)
    ax.annotate("", xy=(-0.3, 1.13), xytext=(0.3, 1.13), arrowprops=style, zorder=6)
    ax.annotate("", xy=(0.3, -1.13), xytext=(-0.3, -1.13), arrowprops=style, zorder=6)
    ax.text(0, 1.2, "crépuscule (+Y)", ha="center", va="bottom", fontsize=7)
    ax.text(0, -1.2, "aube (−Y)", ha="center", va="top", fontsize=7)


def rotation_planete_xz(ax):
    """
    Vue X-Z : on regarde Mercure depuis le côté -Y, donc la face visible est le côté aube.
    L'axe de rotation est vertical ; sur la face visible la surface défile vers +X (vers le Soleil).
    """
    contour = [pe.withStroke(linewidth=2.5, foreground="k")]
    ax.annotate("", xy=(0.55, 0.0), xytext=(-0.55, 0.0), zorder=6,
                arrowprops=dict(arrowstyle="-|>", color="w", lw=1.6, mutation_scale=13,
                                path_effects=contour, shrinkA=0, shrinkB=0))
    ax.text(0, -0.12, "rotation", ha="center", va="top", fontsize=7, color="w", path_effects=contour,
            zorder=6)


def tracer_orbite(axs, tp, P, deb, fin):
    """Deux projections de la trajectoire (axs[0] : X-Y vue du nord, axs[1] : X-Z vue depuis l'aube)."""
    ok = np.isfinite(P).all(axis=1)
    tp, P = tp[ok], P[ok]
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    dans = (tp >= np.datetime64(deb)) & (tp <= np.datetime64(fin))
    mp_x, mp_rho = magnetopause()

    panneaux = [
        (axs[0], x, y, r"$X_{MSO}$ [$R_M$]", r"$Y_{MSO}$ [$R_M$]",
         [(mp_x, mp_rho), (mp_x, -mp_rho)]),
        (axs[1], x, z, r"$X_{MSO}$ [$R_M$]", r"$Z_{MSO}$ [$R_M$]",
         [(mp_x, Z_DIP + mp_rho), (mp_x, Z_DIP - mp_rho)]),
    ]
    for k, (ax, u, v, xl, yl, courbes_mp) in enumerate(panneaux):
        # Mercure : jour (X > 0) clair, nuit sombre
        ax.add_patch(Wedge((0, 0), 1, -90, 90, fc="#d9a65a", ec="none"))
        ax.add_patch(Wedge((0, 0), 1, 90, 270, fc="#5a4630", ec="none"))
        if k == 0:
            rotation_planete_xy(ax)
        elif k == 1:
            rotation_planete_xz(ax)
        for i, (cx, cy) in enumerate(courbes_mp):
            ax.plot(cx, cy, color="tab:orange", lw=0.9, alpha=0.8,
                    label="magnétopause (Winslow+2013)" if (k == 0 and i == 0) else None)
        ax.plot(u, v, color="0.45", lw=1.0, label="trajectoire (fenêtre)" if k == 0 else None)
        if dans.any():
            ax.plot(u[dans], v[dans], color="tab:red", lw=2.6, label="évènement" if k == 0 else None)
        ax.scatter(u[0], v[0], color="tab:blue", s=28, zorder=5, label="début fenêtre" if k == 0 else None)
        ax.scatter(u[-1], v[-1], color="k", s=28, marker="s", zorder=5,
                   label="fin fenêtre" if k == 0 else None)
        fleche_parcours(ax, u, v)
        if k == 0:
            ax.plot([], [], color="k", marker=">", ls="none", ms=5, label="sens de parcours")
        # Limites : trajectoire + planète, avec marge
        u_all = np.r_[u, -1.1, 1.1]
        v_all = np.r_[v, -1.1, 1.1]
        mu = 0.15 * np.ptp(u_all)
        mv = 0.15 * np.ptp(v_all)
        ax.set_xlim(u_all.min() - mu, u_all.max() + mu)
        ax.set_ylim(v_all.min() - mv, v_all.max() + mv)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(alpha=0.3)
        ax.tick_params(direction="in", top=True, right=True)
    # Légende commune sous les vues, pour ne rien masquer dans les axes
    handles, labels = axs[0].get_legend_handles_labels()
    axs[0].figure.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8, frameon=False)
    axs[0].set_title("vue depuis le nord, Soleil à droite\nrotation de Mercure antihoraire", fontsize=8)
    axs[1].set_title("vue depuis l'aube (−Y), Soleil à droite", fontsize=8)


def tracer_position(axs, tp, P, deb, fin):
    """
    Position de MESSENGER en fonction du temps, trois panneaux empilés (axe temps partagé) :
    altitude, latitude planétocentrique et heure locale.
    Heure locale : 12 h au midi (+X), 18 h au crépuscule (+Y), 6 h à l'aube (-Y), 0 h à minuit.
    """
    ok = np.isfinite(P).all(axis=1)
    tp, P = tp[ok], P[ok]
    if len(tp) < 2:
        message_axe(axs[0], "position insuffisante")
        return
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    r = np.linalg.norm(P, axis=1)
    alt = (r - 1.0) * R_MERCURE_KM
    lat = np.degrees(np.arcsin(np.clip(z / r, -1, 1)))
    lt = 12 + np.unwrap(np.arctan2(y, x)) * 12 / np.pi
    lt -= 24 * np.floor(np.median(lt) / 24)  # ramène la série dans [0, 24[ sans saut au passage de minuit

    t = mdates.date2num(tp)
    series = [(alt, "altitude [km]"), (lat, "latitude [°]"), (lt, "heure locale [h]")]
    for i, (ax, (v, lab)) in enumerate(zip(axs, series)):
        ax.plot(t, v, color="0.35", lw=1.2)
        ax.axvspan(deb, fin, color="tab:orange", alpha=0.15, lw=0)
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
        ax.tick_params(direction="in", top=True, right=True)
        if i < 2:
            ax.tick_params(labelbottom=False)
    axs[1].axhline(0, color="k", ls=":", lw=0.8)
    axs[0].set_xlim(t[0], t[-1])
    axs[0].set_title("position de MESSENGER (heure locale : 6 h aube, 12 h midi, 18 h crépuscule)", fontsize=8)
    axs[-1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=5))
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axs[-1].set_xlabel("UTC")


# ======================================================================
# Un évènement = une figure
# ======================================================================
def traiter_evenement(groupe, e, force=False):
    """Construit la figure d'un évènement. Renvoie (fig, texte_de_statut)."""
    deb, fin = parse(e.start), parse(e.end)
    wd, wf = fenetre_elargie(deb, fin)

    if wd < PLAGE_B[0] or wf > PLAGE_B[1]:
        raise DonneesInsuffisantes(
            f"hors de la plage de {PRODUIT_B} ({PLAGE_B[0]:%Y-%m-%d} → {PLAGE_B[1]:%Y-%m-%d}) : "
            "évènement ignoré (survol avant l'insertion en orbite ?)")
    donnees_b = telecharger(PRODUIT_B, wd, wf, force)
    if donnees_b is None:
        raise DonneesInsuffisantes("aucune donnée de champ magnétique AMDA")
    t, B = donnees_b
    manquant = ~np.isfinite(B).all(axis=1)
    if manquant.all():  # rien à tracer : on n'écrit pas de figure vide
        raise DonneesInsuffisantes(
            f"champ magnétique entièrement manquant (NaN) dans AMDA sur la fenêtre ({len(B)} échantillons)")
    notes = []
    if manquant.any():
        dt = np.median(np.diff(t).astype("timedelta64[ns]").astype(float)) * 1e-9
        max_trou = max(1, int(round(MAX_TROU_S / dt))) if dt > 0 else int(MAX_TROU_S * FS_DEFAULT)
        B = combler_trous(B, max_trou)
        n_av, n_ap = int(manquant.sum()), int((~np.isfinite(B).all(axis=1)).sum())
        notes.append(f"{n_av} échantillons manquants ({100 * n_av / len(B):.1f} %), "
                     f"{n_av - n_ap} comblés par la moyenne des voisins")
    donnees_xyz = telecharger(PRODUIT_XYZ, wd, wf, force)

    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.4, 1.3], hspace=0.30, wspace=0.28,
                          left=0.06, right=0.98, top=0.92, bottom=0.10)
    gs_b = gs[0, 0:2].subgridspec(4, 1, hspace=0.08)
    axs_b = [fig.add_subplot(gs_b[0])]
    axs_b += [fig.add_subplot(gs_b[i], sharex=axs_b[0]) for i in range(1, 4)]
    gs_p = gs[0, 2].subgridspec(2, 1, hspace=0.38)
    ax_log, ax_lin = fig.add_subplot(gs_p[0]), fig.add_subplot(gs_p[1])
    axs_o = [fig.add_subplot(gs[1, j]) for j in range(2)]
    gs_pos = gs[1, 2].subgridspec(3, 1, hspace=0.12)
    axs_pos = [fig.add_subplot(gs_pos[0])]
    axs_pos += [fig.add_subplot(gs_pos[i], sharex=axs_pos[0]) for i in (1, 2)]

    tracer_b(axs_b, t, B, deb, fin, wd, wf)

    try:
        r = calcul_psd(t, B)
        tracer_psd(ax_log, ax_lin, r)
        if abs(r["fs"] - FS_DEFAULT) > 1:
            notes.append(f"cadence estimée {r['fs']:.2f} Hz (attendu {FS_DEFAULT:.0f})")
    except DonneesInsuffisantes as exc:
        message_axe(ax_log, str(exc))
        message_axe(ax_lin, "")
        notes.append(str(exc))

    if donnees_xyz is None:
        for ax in axs_o + axs_pos[:1]:
            message_axe(ax, "pas de données de position")
        for ax in axs_pos[1:]:
            ax.axis("off")
        notes.append("pas de position")
    else:
        tracer_orbite(axs_o, donnees_xyz[0], donnees_xyz[1], deb, fin)
        tracer_position(axs_pos, donnees_xyz[0], donnees_xyz[1], deb, fin)

    duree = (fin - deb).total_seconds() / 60
    fig.suptitle(
        f"{groupe} | {deb:%Y-%m-%d %H:%M} → {fin:%Y-%m-%d %H:%M} UTC (durée {duree:.0f} min) | "
        f"fenêtre affichée ×{FACTEUR_FENETRE:g} : {wd:%H:%M:%S} → {wf:%H:%M:%S}",
        fontsize=12)
    return fig, "; ".join(notes)


# ======================================================================
# Programme principal
# ======================================================================
def evenement_manuel(debut, fin, force=True):
    """Traite un seul évènement donné en ligne de commande ; le PNG va dans data/manuel/."""
    e = evmod.Event(debut.strip(), fin.strip())
    nom = nom_evenement(parse(e.start), parse(e.end))
    dossier = DATA / "manuel"
    dossier.mkdir(parents=True, exist_ok=True)
    try:
        fig, notes = traiter_evenement("manuel", e, force)
        chemin = dossier / f"{nom}.png"
        fig.savefig(chemin, dpi=DPI)
        plt.close(fig)
        journal(f"OK      [manuel] {nom} -> {chemin}" + (f"  ({notes})" if notes else ""))
    except DonneesInsuffisantes as exc:
        plt.close("all")
        journal(f"IGNORE  [manuel] {nom}  {exc}")
    except Exception as exc:
        plt.close("all")
        journal(f"ERREUR  [manuel] {nom}  {type(exc).__name__}: {exc}\n{traceback.format_exc()}")


def main():
    ap = argparse.ArgumentParser(description="Figures tout-en-un pour les évènements KH de events.py")
    ap.add_argument("--evenement", nargs=2, metavar=("DEBUT", "FIN"),
                    help='un seul évènement, dates "YYYY-MM-DD HH:MM" ; PNG dans data/manuel/')
    ap.add_argument("--groupes", nargs="+", choices=list(GROUPES), default=list(GROUPES),
                    help="groupes à traiter (défaut : tous)")
    ap.add_argument("--limite", type=int, default=None, help="n premiers évènements de chaque groupe")
    ap.add_argument("--no-force", dest="force", action="store_false",
                    help="réutilise le cache s'il existe (par défaut : réimport systématique depuis AMDA)")
    ap.add_argument("--pdf", action="store_true", help="PDF récapitulatif par groupe en plus des PNG")
    args = ap.parse_args()

    if args.evenement:
        try:
            deb, fin = (parse(s) for s in args.evenement)
        except ValueError:
            ap.error('dates attendues au format "YYYY-MM-DD HH:MM", ex. --evenement "2011-05-15 09:26" "2011-05-15 09:31"')
        if fin <= deb:
            ap.error("la fin doit être postérieure au début")
        evenement_manuel(*args.evenement, force=args.force)
        return

    bilan = dict(ok=0, ignore=0, erreur=0)
    for groupe in args.groupes:
        liste = GROUPES[groupe][: args.limite]
        dossier = DATA / groupe
        dossier.mkdir(parents=True, exist_ok=True)
        pdf = PdfPages(dossier / f"{groupe}_recap.pdf") if args.pdf else None
        journal(f"=== {groupe} : {len(liste)} évènements ===")
        try:
            for i, e in enumerate(liste, 1):
                nom = nom_evenement(parse(e.start), parse(e.end))
                etiquette = f"[{groupe} {i}/{len(liste)}] {nom}"
                try:
                    fig, notes = traiter_evenement(groupe, e, args.force)
                    fig.savefig(dossier / f"{nom}.png", dpi=DPI)
                    if pdf is not None:
                        pdf.savefig(fig)
                    plt.close(fig)
                    bilan["ok"] += 1
                    journal(f"OK      {etiquette}" + (f"  ({notes})" if notes else ""))
                except DonneesInsuffisantes as exc:
                    plt.close("all")
                    bilan["ignore"] += 1
                    journal(f"IGNORE  {etiquette}  {exc}")
                except Exception as exc:  # une erreur sur un évènement n'arrête pas la boucle
                    plt.close("all")
                    bilan["erreur"] += 1
                    journal(f"ERREUR  {etiquette}  {type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        finally:
            if pdf is not None:
                pdf.close()
    journal(f"Bilan : {bilan['ok']} figures, {bilan['ignore']} ignorés, {bilan['erreur']} erreurs")


if __name__ == "__main__":
    main()
