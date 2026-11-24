#!/usr/bin/env python3
"""
webapp.py : exploration interactive des données MESSENGER (AMDA via speasy), dans le navigateur.

Reprend la sortie de traitement.py (B, PSD, orbite, position) en figures Plotly zoomables, et permet de
sélectionner une zone temporelle sur les courbes de B pour en calculer la PSD.

Tous les calculs physiques viennent de traitement.py (téléchargement + cache data/cache/, comblement des trous,
calcul_psd selon Gershman et al. 2015, fréquences cyclotron, magnétopause de Winslow) : une modification de
traitement.py se répercute donc ici. Ce fichier ajoute l'interface, le lissage d'affichage de la PSD et
l'ajustement des pentes.

Lancement (depuis la racine du dépôt) :
  pip install dash            (en plus de numpy, scipy, matplotlib, speasy)
  python webapp.py            puis ouvrir http://127.0.0.1:8050
  python webapp.py --port 8051 --debug
  python webapp.py --host 0.0.0.0   accessible depuis les autres machines (réseau local, Tailscale), sans --debug

Utilisation :
  1. Choisir un évènement de events.py ou saisir début / fin ("YYYY-MM-DD HH:MM[:SS]"), puis « Charger ».
     Comme dans traitement.py, les données sont chargées sur la fenêtre FACTEUR x la durée de l'évènement,
     centrée sur celui-ci (augmenter le facteur pour voir les tendances à plus grande échelle).
  2. Molette / glisser pour zoomer, outil « Pan » de la barre Plotly pour se déplacer, double-clic ou
     « ↺ fenêtre initiale » pour revenir (temps et bornes de B). Les bornes de chaque panneau de B suivent
     le min / max de la zone affichée (+ 8 %) à chaque zoom ou déplacement, tant que la case « bornes de B = min / max »
     est cochée. Pour une grande fenêtre, la zone affichée est tracée à pleine
     résolution et le reste réduit par min-max (les pics restent visibles), raffiné dès qu'on y zoome.
  3. PSD : celle de toute la fenêtre est toujours affichée à droite. Pour une autre zone : outil « Box Select »
     sur les courbes de B (calcul automatique), bornes saisies à la main, zone affichée, ou évènement seul.
  4. Réglages PSD (s'appliquent aux deux PSD) :
     - ions dont on trace la fréquence cyclotron ;
     - lissage : moyenne glissante de log10(PSD) sur une largeur fixe en décades (affichage seulement) ;
     - pentes automatiques : ajustement affine par morceaux (continu) de log10(PSD) en fonction de log10(f)
       sur [f min, f max] (0,1-10 Hz par défaut), après regroupement en intervalles réguliers en log(f) pour
       ne pas surpondérer les hautes fréquences. Nombre de ruptures fixé (1 à 3) ou choisi par critère BIC ;
     - pentes manuelles : cocher « pente manuelle » puis cliquer deux points sur le panneau log-log.
  5. Notes et catégories (en haut) : texte libre sur l'évènement, et étiquettes créées à la volée
     (« KH/flanc aube » crée un sous-dossier « flanc aube » dans « KH » ; un évènement peut en avoir plusieurs).
  6. « Sauver dans le projet » écrit tout l'état de l'écran dans data/JSONS/<dates>.json (un fichier par
     évènement, remplacé à chaque sauvegarde) : intervalle, zoom, sélection, réglages, pentes manuelles, notes,
     catégories, et dans "resultats" toutes les valeurs affichées (caractéristiques des PSD, fréquences
     cyclotron, pentes automatiques avec ruptures et écarts-types, pentes manuelles avec leur valeur).
     « Exporter » télécharge ce même JSON ; « Importer » en ouvre un depuis l'ordinateur.
  7. L'explorateur à gauche (repliable avec ⟨) liste data/JSONS, en 📁 dossiers (un par catégorie ; un évènement
     apparaît dans chacune de ses catégories) ou en 🏷 étiquettes (puces cumulables : seuls les évènements qui les
     ont toutes restent). Un clic rouvre l'évènement dans l'état sauvegardé ; ◀ / ▶ passent au précédent /
     suivant de la liste affichée. Le menu events.py sert à démarrer l'étude d'un nouvel évènement (✓ = déjà
     sauvegardé ; ses notes et catégories sont alors reprises).
     « 🗑 retirer » enlève l'évènement ouvert de la liste : son JSON est déplacé dans data/JSONS/_corbeille/.
     Une catégorie désuète se supprime dans le cadre « Catégories » (elle est retirée de tous les JSON).
  8. À plusieurs : l'explorateur relit data/JSONS toutes les 10 s, et « Sauver dans le projet » refuse d'écraser
     sans confirmation un JSON modifié par quelqu'un d'autre depuis qu'on l'a ouvert.
"""
import argparse
import base64
import itertools
import json
import re
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update, Patch

import traitement as tr

# ======================================================================
# Configuration de l'application
# ======================================================================
MEMOIRE = OrderedDict()   # clé -> dict(t, B, deb, fin, wd, wf, facteur, xyz, notes)
MEMOIRE_MAX = 6           # nombre de chargements gardés en mémoire
CACHE_PSD = OrderedDict()  # (clé, sélection) -> résultat de tr.calcul_psd ou message d'erreur
CACHE_AJUST = OrderedDict()

# Affichage de B : pleine résolution dans la zone visible (min-max au-delà de NB_VISIBLE intervalles),
# min-max sur NB_HORS intervalles pour le reste de la fenêtre
NB_VISIBLE = 6000
NB_HORS = 1500
MAX_POINTS_ORBITE = 6000
Y_MARGE = 0.08    # marge relative autour du min / max de B dans la zone affichée

FIT_FMIN, FIT_FMAX = 0.1, 10.0   # Hz, plage par défaut de l'ajustement des pentes
FIT_NB_BINS = 40                 # intervalles réguliers en log10(f) pour l'ajustement
FIT_MIN_BINS = 5                 # intervalles minimum par segment (5 sur 40 = 0,25 décade sur 0,1-10 Hz)
VERSION_CONFIG = 4

# Ions proposés pour les fréquences cyclotron : ceux de traitement.py, plus d'autres ions de l'environnement
# de Mercure (exosphère, vent solaire)
IONS_DISPO = dict(tr.IONS)
IONS_DISPO.update({"He+": (4, 1), "Mg+": (24, 1), "K+": (39, 1), "Ca+": (40, 1)})
COULEURS_IONS = dict(tr.COULEURS_IONS)
COULEURS_IONS.update({"He+": "olive", "Mg+": "#17becf", "K+": "#8c564b", "Ca+": "#e377c2"})

COUL_EVT = "rgba(255,127,14,0.22)"   # orange de traitement.py : évènement
COUL_SEL = "rgba(128,0,128,0.15)"    # violet : zone sélectionnée
COMPOSANTES = {"perp": ("P⊥", "purple", "#4b0055"), "para": ("P∥", "mediumseagreen", "#0b6b3a")}
COUL_MANUELLE = "#d62728"
CONFIG_FIG = {"scrollZoom": True, "displaylogo": False,
              "toImageButtonOptions": {"format": "png", "scale": 2}}


# Évènements étudiés : un JSON par évènement dans data/JSONS (nom tiré des dates), classés par étiquettes
DOSSIER_JSONS = tr.DATA / "JSONS"
FICHIER_CATEGORIES = DOSSIER_JSONS / "_categories.json"   # catégories créées (même vides)
CORBEILLE = DOSSIER_JSONS / "_corbeille"   # JSON retirés de la liste (récupérables à la main)
SONDAGE_S = 10   # relecture de data/JSONS (sauvegardes d'un camarade) toutes les SONDAGE_S secondes


def parse_date(s):
    """Accepte "YYYY-MM-DD HH:MM", "YYYY-MM-DD HH:MM:SS[.ffffff]" et le séparateur T (format Plotly inclus)."""
    s = str(s or "").strip().replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise ValueError(f"date illisible : « {s} » (format attendu YYYY-MM-DD HH:MM[:SS])")


def fmt_date(d, ms=False):
    return f"{d:%Y-%m-%d %H:%M:%S.%f}"[:-3] if ms else f"{d:%Y-%m-%d %H:%M:%S}"


def fmt_num(v, defaut):
    try:
        return float(v)
    except (TypeError, ValueError):
        return defaut


# ======================================================================
# Données
# ======================================================================
def charger(deb, fin, facteur, force):
    """
    Télécharge (ou relit le cache) B et la position sur la fenêtre FACTEUR x la durée de l'évènement, centrée
    sur celui-ci (comme traitement.py) ; comble les petits trous. Renvoie la clé en mémoire.
    """
    marge = (fin - deb) * ((facteur - 1) / 2)   # même formule que tr.fenetre_elargie, facteur réglable
    wd, wf = (deb - marge).replace(microsecond=0), (fin + marge).replace(microsecond=0)
    if wd < tr.PLAGE_B[0] or wf > tr.PLAGE_B[1]:
        raise tr.DonneesInsuffisantes(
            f"fenêtre {fmt_date(wd)} → {fmt_date(wf)} hors de la plage de {tr.PRODUIT_B} "
            f"({tr.PLAGE_B[0]:%Y-%m-%d} → {tr.PLAGE_B[1]:%Y-%m-%d})")

    donnees_b = tr.telecharger(tr.PRODUIT_B, wd, wf, force)
    if donnees_b is None:
        raise tr.DonneesInsuffisantes("aucune donnée de champ magnétique AMDA sur cette fenêtre")
    t, B = donnees_b
    t = t.astype("datetime64[ns]")
    manquant = ~np.isfinite(B).all(axis=1)
    if manquant.all():
        raise tr.DonneesInsuffisantes(f"champ magnétique entièrement manquant (NaN) ({len(B)} échantillons)")
    notes = [f"{len(B)} échantillons"]
    if manquant.any():
        dt = np.median(np.diff(t).astype("timedelta64[ns]").astype(float)) * 1e-9
        max_trou = max(1, int(round(tr.MAX_TROU_S / dt))) if dt > 0 else int(tr.MAX_TROU_S * tr.FS_DEFAULT)
        B = tr.combler_trous(B, max_trou)
        n_av, n_ap = int(manquant.sum()), int((~np.isfinite(B).all(axis=1)).sum())
        notes.append(f"{n_av} manquants ({100 * n_av / len(B):.1f} %), {n_av - n_ap} comblés")
    xyz = tr.telecharger(tr.PRODUIT_XYZ, wd, wf, force)
    if xyz is None:
        notes.append("pas de position")
    else:
        xyz = (xyz[0].astype("datetime64[ns]"), xyz[1])

    cle = f"{wd:%Y%m%dT%H%M%S}_{wf:%Y%m%dT%H%M%S}_{datetime.now():%H%M%S%f}"
    MEMOIRE[cle] = dict(t=t, t_ms=t.astype("datetime64[ms]").astype(np.int64), B=B,
                        B_abs=np.linalg.norm(B, axis=1), deb=deb, fin=fin, wd=wd, wf=wf,
                        facteur=facteur, xyz=xyz, notes=notes)
    while len(MEMOIRE) > MEMOIRE_MAX:
        MEMOIRE.popitem(last=False)
    return cle


def psd(cle, sel=None):
    """PSD de toute la fenêtre (sel=None) ou d'une sélection [début, fin] ; résultat mis en cache."""
    k = (cle, tuple(sel) if sel else None)
    if k not in CACHE_PSD:
        d = MEMOIRE[cle]
        a, b = (parse_date(sel[0]), parse_date(sel[1])) if sel else (d["wd"], d["wf"])
        m = (d["t"] >= np.datetime64(a)) & (d["t"] <= np.datetime64(b))
        try:
            CACHE_PSD[k] = tr.calcul_psd(d["t"][m], d["B"][m])
        except tr.DonneesInsuffisantes as exc:
            CACHE_PSD[k] = str(exc)
        while len(CACHE_PSD) > 30:
            CACHE_PSD.popitem(last=False)
    return CACHE_PSD[k]


# ======================================================================
# Réduction pour l'affichage de B
# ======================================================================
def minmax(x, y, nb):
    """Garde, dans chacun des nb intervalles, le minimum et le maximum de y (dans l'ordre) : les pics restent."""
    n = len(y)
    if n <= 2 * nb:
        return x, y
    bords = np.linspace(0, n, nb + 1).astype(int)
    lo = np.where(np.isfinite(y), y, np.inf)
    hi = np.where(np.isfinite(y), y, -np.inf)
    i_min = np.array([a + np.argmin(lo[a:b]) for a, b in zip(bords[:-1], bords[1:])])
    i_max = np.array([a + np.argmax(hi[a:b]) for a, b in zip(bords[:-1], bords[1:])])
    idx = np.unique(np.r_[i_min, i_max])
    return x[idx], y[idx]


def plages_b(d, a, b):
    """
    Bornes verticales des 4 panneaux de B : minimum et maximum des données entre a et b (la zone affichée),
    plus une marge de Y_MARGE. None si pas assez de données.
    """
    i0 = np.searchsorted(d["t_ms"], np.datetime64(a, "ms").astype(np.int64))
    i1 = np.searchsorted(d["t_ms"], np.datetime64(b, "ms").astype(np.int64), side="right")
    plages = []
    for y in (d["B"][:, 0], d["B"][:, 1], d["B"][:, 2], d["B_abs"]):
        v = y[i0:i1]
        v = v[np.isfinite(v)]
        if len(v) < 2:
            plages.append(None)
            continue
        lo, hi = v.min(), v.max()
        marge = max(Y_MARGE * (hi - lo), 1.0)
        plages.append([float(lo - marge), float(hi + marge)])
    return plages


def bornes_vue(d, vue):
    return (parse_date(vue[0]), parse_date(vue[1])) if vue else (d["wd"], d["wf"])


def donnees_affichees(d, vue):
    """
    x, y pour Bx, By, Bz, |B| : pleine résolution dans la vue, min-max ailleurs.
    x est envoyé en texte ISO sans fuseau ("2011-05-15T09:23:30.050") : Plotly lit les dates en texte telles
    quelles (UTC ici), alors qu'il convertirait des millisecondes numériques dans le fuseau du navigateur
    (décalage d'une ou deux heures en France).
    """
    a, b = (parse_date(vue[0]), parse_date(vue[1])) if vue else (d["wd"], d["wf"])
    tms = d["t_ms"]
    i0 = np.searchsorted(tms, np.datetime64(a, "ms").astype(np.int64))
    i1 = np.searchsorted(tms, np.datetime64(b, "ms").astype(np.int64), side="right")
    n = len(tms)
    nb_g = max(1, int(NB_HORS * i0 / max(1, n - (i1 - i0))))
    nb_d = max(1, NB_HORS - nb_g)
    sorties = []
    for y in (d["B"][:, 0], d["B"][:, 1], d["B"][:, 2], d["B_abs"]):
        morceaux = [minmax(tms[s], y[s], nb) for s, nb in ((slice(0, i0), nb_g), (slice(i0, i1), NB_VISIBLE),
                                                             (slice(i1, n), nb_d))]
        x = np.concatenate([m[0] for m in morceaux]).astype("datetime64[ms]")
        sorties.append((np.datetime_as_string(x, unit="ms"), np.concatenate([m[1] for m in morceaux])))
    return sorties


# ======================================================================
# Traitements de la PSD : lissage, pentes
# ======================================================================
def lisser(f, P, largeur_dec):
    """
    Moyenne glissante de log10(P) sur une fenêtre de largeur constante en décades, centrée sur chaque fréquence.
    La largeur relative est la même à toutes les fréquences, ce qui convient à la lecture des pentes en log-log.
    Le bin f = 0 est laissé tel quel.
    """
    if not largeur_dec or largeur_dec <= 0:
        return P
    x = np.log10(f[1:])
    ly = np.log10(np.maximum(P[1:], np.finfo(float).tiny))
    cs = np.r_[0.0, np.cumsum(ly)]
    lo = np.searchsorted(x, x - largeur_dec / 2, side="left")
    hi = np.searchsorted(x, x + largeur_dec / 2, side="right")
    return np.r_[P[0], 10 ** ((cs[hi] - cs[lo]) / (hi - lo))]


def matrice_ext(xx, br):
    xx = np.asarray(xx, dtype=float)
    return np.column_stack([np.ones(len(xx)), xx] + [np.maximum(0.0, xx - b) for b in br])


def ajuster_pentes(f, P, fmin, fmax, mode):
    """
    Ajustement affine par morceaux, continu, de y = log10(P) en fonction de x = log10(f) sur [fmin, fmax].
    Les points sont d'abord moyennés dans FIT_NB_BINS intervalles réguliers en x (sinon les hautes fréquences,
    bien plus nombreuses en échelle linéaire, dicteraient l'ajustement). Modèle :
        y = a + b x + sum_i c_i max(0, x - x_i)
    Les ruptures x_i sont cherchées exhaustivement parmi les milieux d'intervalles (au moins FIT_MIN_BINS
    intervalles par segment), puis affinées (Nelder-Mead, même contrainte). mode = "auto" : nombre de ruptures
    (0 à 3) choisi par le critère BIC ; mode = "1", "2" ou "3" : nombre imposé.
    Renvoie un dict (ruptures en Hz, pentes et écarts-types, fonction modèle) ; lève ValueError si impossible.
    """
    m = (f > 0) & (f >= fmin) & (f <= fmax) & np.isfinite(P) & (P > 0)
    if m.sum() < 2 * FIT_MIN_BINS:
        raise ValueError(f"trop peu de points entre {fmin:g} et {fmax:g} Hz")
    x, y = np.log10(f[m]), np.log10(P[m])
    bords = np.linspace(x.min(), x.max(), FIT_NB_BINS + 1)
    idx = np.clip(np.digitize(x, bords) - 1, 0, FIT_NB_BINS - 1)
    occupes = [i for i in range(FIT_NB_BINS) if (idx == i).any()]
    xb = np.array([x[idx == i].mean() for i in occupes])
    yb = np.array([y[idx == i].mean() for i in occupes])
    n = len(xb)
    largeur_min = 0.9 * FIT_MIN_BINS * (x.max() - x.min()) / FIT_NB_BINS

    def rss(br):
        X = matrice_ext(xb, br)
        beta = np.linalg.lstsq(X, yb, rcond=None)[0]
        r = yb - X @ beta
        return float(r @ r), beta

    def cout(v):  # l'affinage respecte aussi la largeur minimale des segments
        v = np.sort(v)
        if np.diff(np.r_[x.min(), v, x.max()]).min() < largeur_min:
            return np.inf
        return rss(v)[0]

    def meilleur(k):
        if k == 0:
            return [], rss([])[0]
        positions = range(FIT_MIN_BINS, n - FIT_MIN_BINS + 1)
        best = (np.inf, None)
        for combi in itertools.combinations(positions, k):
            if any(b - a < FIT_MIN_BINS for a, b in zip(combi, combi[1:])):
                continue
            br = [(xb[i - 1] + xb[i]) / 2 for i in combi]
            r = rss(br)[0]
            if r < best[0]:
                best = (r, br)
        if best[1] is None:
            return None, np.inf
        opt = minimize(cout, best[1], method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-10))
        br = list(np.sort(opt.x))
        r = cout(br)
        return (br, r) if r <= best[0] else (best[1], best[0])

    k_max = max(0, min(3, n // FIT_MIN_BINS - 1))
    candidats = range(0, k_max + 1) if mode == "auto" else [min(int(mode), k_max)]
    resultats = []
    for k in candidats:
        br, r = meilleur(k)
        if br is None:
            continue
        bic = n * np.log(max(r, 1e-300) / n) + (2 + 2 * k) * np.log(n)
        resultats.append((bic, k, br, r))
    if not resultats:
        raise ValueError("ajustement impossible")
    _, k, br, r = min(resultats, key=lambda e: e[0])

    X = matrice_ext(xb, br)
    beta = np.linalg.lstsq(X, yb, rcond=None)[0]
    p = 2 + 2 * k
    s2 = r / (n - p) if n > p else r / n
    cov = s2 * np.linalg.pinv(X.T @ X)
    bornes = np.r_[x.min(), br, x.max()]
    segments = []
    for j in range(k + 1):
        L = np.zeros(len(beta))
        L[1] = 1
        L[2:2 + j] = 1
        segments.append(dict(f1=10 ** bornes[j], f2=10 ** bornes[j + 1], pente=float(L @ beta),
                             err=float(np.sqrt(max(L @ cov @ L, 0)))))

    def modele(xx):
        return matrice_ext(xx, br) @ beta

    return dict(k=k, auto=(mode == "auto"), ruptures=[10 ** b for b in br], segments=segments,
                modele=modele, xmin=x.min(), xmax=x.max(), n_bins=n)


def ajustement_cache(cle_cache, f, P, fmin, fmax, mode):
    """ajuster_pentes mémorisé : le curseur de lissage ne relance pas l'ajustement."""
    if cle_cache not in CACHE_AJUST:
        try:
            CACHE_AJUST[cle_cache] = ajuster_pentes(f, P, fmin, fmax, mode)
        except ValueError as exc:
            CACHE_AJUST[cle_cache] = exc
        while len(CACHE_AJUST) > 60:
            CACHE_AJUST.popitem(last=False)
    res = CACHE_AJUST[cle_cache]
    if isinstance(res, Exception):
        raise res
    return res


def pente_manuelle(p):
    return np.log10(p["p2"] / p["p1"]) / np.log10(p["f2"] / p["f1"])


def composantes_choisies(reglages):
    c = reglages.get("composante") or "perp"
    return ["perp", "para"] if c == "both" else [c]


def ions_choisis(reglages):
    ions = reglages.get("ions")
    return [i for i in (list(tr.IONS) if ions is None else ions) if i in IONS_DISPO]


# ======================================================================
# Figures
# ======================================================================
LIGNES_B = [("x", "y"), ("x2", "y2"), ("x3", "y3"), ("x4", "y4")]


def vrects(deb, fin, couleur, lignes):
    """Bandes verticales (une par sous-graphe) ; lignes = liste des (xref, yref)."""
    # line=dict(width=0) et non line_width=0 : ces formes passent aussi par des Patch, où Plotly ne
    # traduit pas la notation abrégée (le contour par défaut réapparaîtrait)
    return [dict(type="rect", xref=xr, yref=f"{yr} domain", x0=deb, x1=fin, y0=0, y1=1,
                 fillcolor=couleur, line=dict(width=0), layer="below") for xr, yr in lignes]


def formes_b(d, sel=None):
    """Bandes évènement / sélection et lignes B = 0 des quatre sous-graphes de B."""
    formes = [dict(type="line", xref=f"{xr} domain", yref=yr, x0=0, x1=1, y0=0, y1=0,
                   line=dict(color="black", dash="dot", width=0.8)) for xr, yr in LIGNES_B[:3]]
    formes += vrects(d["deb"], d["fin"], COUL_EVT, LIGNES_B)
    if sel:
        formes += vrects(sel[0], sel[1], COUL_SEL, LIGNES_B)
    return formes


def fig_b(cle, vue=None, sel=None):
    d = MEMOIRE[cle]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.025)
    for i, ((x, y), nom) in enumerate(zip(donnees_affichees(d, vue), ("Bx", "By", "Bz", "|B|")), 1):
        # Marqueurs invisibles : nécessaires pour que l'outil de sélection Plotly renvoie la zone choisie.
        fig.add_trace(go.Scattergl(x=x, y=y, mode="lines+markers", name=nom, showlegend=False,
                                   line=dict(color="black", width=1),
                                   marker=dict(size=3, opacity=0),
                                   selected=dict(marker=dict(opacity=0)),
                                   unselected=dict(marker=dict(opacity=0)),
                                   hovertemplate="%{x|%H:%M:%S.%L}<br>" + nom + " = %{y:.2f} nT<extra></extra>"),
                      row=i, col=1)
        fig.update_yaxes(title_text=f"{nom} (nT)", row=i, col=1)
    duree = (d["fin"] - d["deb"]).total_seconds() / 60
    titre = (f"évènement {d['deb']:%Y-%m-%d %H:%M:%S} → {d['fin']:%H:%M:%S} UTC (durée {duree:.1f} min) — "
             f"fenêtre ×{d['facteur']:g} : {d['wd']:%H:%M:%S} → {d['wf']:%H:%M:%S}")
    s = (parse_date(sel[0]), parse_date(sel[1])) if sel else None
    fig.update_layout(shapes=formes_b(d, s), title=dict(text=titre, font_size=12),
                      height=620, margin=dict(l=60, r=15, t=45, b=40),
                      dragmode="zoom", selectdirection="h", hovermode="x",
                      uirevision=cle, template="plotly_white")
    # La plage initiale (celle que rétablit le double-clic) est la fenêtre ×facteur
    x_range = list(bornes_vue(d, vue))
    fig.update_xaxes(type="date", range=x_range, showspikes=True, spikemode="across", spikethickness=1)
    # Bornes de B calées sur la zone affichée (sinon l'échelle couvre toute l'orbite, périherme compris)
    for i, plage in enumerate(plages_b(d, *x_range), 1):
        if plage:
            fig.update_yaxes(range=plage, autorange=False, row=i, col=1)
    fig.update_xaxes(title_text="UTC", tickformat="%H:%M:%S\n%Y-%m-%d", row=4, col=1)
    return fig


def fig_vide(texte, height=300):
    fig = go.Figure()
    fig.add_annotation(text=texte, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font=dict(color="firebrick", size=13))
    fig.update_layout(height=height, template="plotly_white", xaxis_visible=False, yaxis_visible=False,
                      margin=dict(l=20, r=20, t=30, b=20))
    return fig


def refs(fig, row, col):
    sp = fig.get_subplot(row, col)
    return sp.xaxis.plotly_name.replace("axis", ""), sp.yaxis.plotly_name.replace("axis", "")


def tracer_psd_panneau(fig, r, row, col, log_x, legende, lissage, ions):
    """Un panneau de PSD : log-log (log_x=True) ou zoom en fréquence linéaire jusqu'à ZOOM_LIN."""
    f = r["f"]
    m = np.arange(1, len(f)) if log_x else np.flatnonzero((f > 0) & (f <= tr.ZOOM_LIN))
    for comp, (nom, coul, _) in COMPOSANTES.items():
        brut = r[comp]
        lisse = lisser(f, brut, lissage)
        # customdata "log" : seuls les clics sur le panneau log-log servent aux pentes manuelles
        cd = ["log" if log_x else "lin"] * len(m)
        if lissage:
            fig.add_trace(go.Scatter(x=f[m], y=brut[m], mode="lines", name=nom + " brut", legendgroup=nom,
                                     showlegend=False, line=dict(color=coul, width=0.8), opacity=0.25,
                                     customdata=cd, hoverinfo="skip"), row=row, col=col)
        fig.add_trace(go.Scatter(x=f[m], y=lisse[m], mode="lines" if log_x else "lines+markers", name=nom,
                                 legendgroup=nom, showlegend=legende, customdata=cd,
                                 line=dict(color=coul, width=1.5), marker=dict(size=4),
                                 hovertemplate="f = %{x:.4g} Hz<br>" + nom + " = %{y:.3g} nT²/Hz<extra></extra>"),
                      row=row, col=col)

    f_min, f_max = (f[1], f[-1]) if log_x else (0, tr.ZOOM_LIN)
    xd, yd = refs(fig, row, col)
    if log_x:  # bins atténués par le detrending
        fig.add_shape(type="rect", xref=xd, yref=f"{yd} domain", x0=f[1], x1=2 * f[1], y0=0, y1=1,
                      fillcolor="rgba(0,0,0,0.08)", line_width=0, layer="below")
    for ion in ions:
        A, q = IONS_DISPO[ion]
        fc = tr.cyclo(A, r["B_moy"], q)
        if not f_min < fc < f_max:
            continue
        coul = COULEURS_IONS[ion]
        fig.add_shape(type="rect", xref=xd, yref=f"{yd} domain",
                      x0=tr.cyclo(A, r["B_moy"] - r["B_std"], q), x1=tr.cyclo(A, r["B_moy"] + r["B_std"], q),
                      y0=0, y1=1, fillcolor=coul, opacity=0.15, line_width=0, layer="below")
        fig.add_shape(type="line", xref=xd, yref=f"{yd} domain", x0=fc, x1=fc, y0=0, y1=1,
                      line=dict(color=coul, dash="dash", width=1.2))
        # Les annotations sur un axe log se placent en log10(x)
        fig.add_annotation(x=np.log10(fc) if log_x else fc, y=1, xref=xd, yref=f"{yd} domain",
                           text=f"{ion} : {fc:.3g} Hz", textangle=-90, showarrow=False,
                           xanchor="right", yanchor="top", font=dict(size=10, color=coul))

    if log_x:
        # Barre d'IC à 95 % (lissage sur 3 bins, fenêtre rectangulaire : 6 ddl), comme traitement.py
        nu = 6
        ic_bas, ic_haut = nu / chi2.ppf(0.975, nu), nu / chi2.ppf(0.025, nu)
        y_ok = np.r_[r["perp"][1:], r["para"][1:]]
        y_ok = y_ok[np.isfinite(y_ok) & (y_ok > 0)]
        if len(y_ok):
            y_bas, y_haut = y_ok.min(), y_ok.max()
            x_ic, y_ic = f[-1] / 2, y_haut / (y_haut / y_bas) ** 0.3
            fig.add_trace(go.Scatter(x=[x_ic], y=[y_ic * ic_haut * 1.6], mode="text", text=["IC 95 %"],
                                     showlegend=False, hoverinfo="skip", textfont=dict(size=10)),
                          row=row, col=col)
            fig.add_trace(go.Scatter(x=[x_ic], y=[y_ic], mode="markers", showlegend=False, hoverinfo="skip",
                                     marker=dict(color="black", size=5),
                                     error_y=dict(type="data", symmetric=False,
                                                  array=[y_ic * (ic_haut - 1)], arrayminus=[y_ic * (1 - ic_bas)],
                                                  color="black", thickness=1.2, width=4)),
                          row=row, col=col)
        fig.update_xaxes(type="log", range=[np.log10(f[1]), np.log10(f[-1])], row=row, col=col)
    else:
        fig.update_xaxes(range=[0, tr.ZOOM_LIN], row=row, col=col)
    fig.update_yaxes(type="log", title_text="PSD [nT²/Hz]", exponentformat="power", row=row, col=col)
    fig.update_xaxes(title_text="Fréquence [Hz]", row=row, col=col)


def tracer_ajustement(fig, ajust, comp, row, col, legende):
    """Droites ajustées, ruptures et valeurs des pentes sur le panneau log-log."""
    nom, _, coul = COMPOSANTES[comp]
    xd, yd = refs(fig, row, col)
    xx = np.linspace(ajust["xmin"], ajust["xmax"], 300)
    yy = ajust["modele"](xx)
    etiquette = f"ajustement {nom} ({ajust['k']} rupture{'s' if ajust['k'] > 1 else ''}"
    etiquette += ", BIC)" if ajust["auto"] else ")"
    fig.add_trace(go.Scatter(x=10 ** xx, y=10 ** yy, mode="lines", name=etiquette, showlegend=legende,
                             line=dict(color=coul, width=2.4, dash="dash"), hoverinfo="skip"), row=row, col=col)
    decal = 2.5 if comp == "perp" else 1 / 2.5   # P⊥ au-dessus de la droite, P∥ en dessous
    for s in ajust["segments"]:
        xm = (np.log10(s["f1"]) + np.log10(s["f2"])) / 2
        ym = ajust["modele"]([xm])[0]
        fig.add_trace(go.Scatter(x=[10 ** xm], y=[10 ** ym * decal], mode="text", showlegend=False,
                                 text=[f"<b>{s['pente']:.2f}</b>"], textfont=dict(color=coul, size=12),
                                 hovertext=f"{nom} : pente {s['pente']:.2f} ± {s['err']:.2f}<br>"
                                           f"{s['f1']:.3g} – {s['f2']:.3g} Hz", hoverinfo="text"),
                      row=row, col=col)
    for fb in ajust["ruptures"]:
        fig.add_shape(type="line", xref=xd, yref=f"{yd} domain", x0=fb, x1=fb, y0=0, y1=0.88,
                      line=dict(color=coul, dash="dot", width=1.3))
        fig.add_trace(go.Scatter(x=[fb], y=[10 ** ajust["modele"]([np.log10(fb)])[0]], mode="markers",
                                 marker=dict(color=coul, size=9, symbol="diamond"), showlegend=False,
                                 hovertemplate=f"rupture {nom} : {fb:.3g} Hz<extra></extra>"), row=row, col=col)


def tracer_manuelles(fig, pentes, attente, row, col):
    for i, p in enumerate(pentes, 1):
        a = pente_manuelle(p)
        fig.add_trace(go.Scatter(x=[p["f1"], p["f2"]], y=[p["p1"], p["p2"]], mode="lines+markers",
                                 line=dict(color=COUL_MANUELLE, width=2), marker=dict(size=7),
                                 name="pentes manuelles", legendgroup="man", showlegend=(i == 1),
                                 hovertemplate=f"m{i} : pente {a:.2f}<extra></extra>"), row=row, col=col)
        fm, pm = np.sqrt(p["f1"] * p["f2"]), np.sqrt(p["p1"] * p["p2"])
        fig.add_trace(go.Scatter(x=[fm], y=[pm / 2.5], mode="text", text=[f"m{i} : {a:.2f}"], showlegend=False,
                                 textfont=dict(color=COUL_MANUELLE, size=12), hoverinfo="skip"), row=row, col=col)
    if attente:
        fig.add_trace(go.Scatter(x=[attente["f"]], y=[attente["p"]], mode="markers", showlegend=False,
                                 marker=dict(color=COUL_MANUELLE, size=13, symbol="x-thin-open",
                                             line=dict(width=3)),
                                 hovertemplate="1er point de la pente : cliquer le 2nd<extra></extra>"),
                      row=row, col=col)


def titre_psd(d, sel, cible):
    if cible == "fen":
        return f"PSD de la fenêtre ×{d['facteur']:g} ({d['wd']:%H:%M:%S} → {d['wf']:%H:%M:%S})"
    a, b = parse_date(sel[0]), parse_date(sel[1])
    nature = " = évènement" if (a, b) == (d["deb"], d["fin"]) else ""
    return f"PSD sélection{nature} : {fmt_date(a)} → {b:%H:%M:%S} UTC ({(b - a).total_seconds():.1f} s)"


def construire_psd(cle, sel, cible, reglages, pentes, attente):
    """Figure de PSD + résumé des pentes pour la fenêtre (cible "fen") ou la sélection (cible "sel")."""
    horizontal = cible == "sel"
    height = 500 if horizontal else 620
    if not cle or cle not in MEMOIRE:
        return fig_vide("Choisir un intervalle puis « Charger »", height), ""
    if cible == "sel" and not sel:
        return fig_vide("Aucune zone sélectionnée", height), ""
    r = psd(cle, sel if cible == "sel" else None)
    titre = titre_psd(MEMOIRE[cle], sel, cible)
    if isinstance(r, str):
        return fig_vide(f"{titre}<br>{r}", height), ""

    sous_titres = [f"⟨|B|⟩ = {r['B_moy']:.1f} ± {r['B_std']:.1f} nT, fs = {r['fs']:.1f} Hz, N = {r['N']}",
                   f"zoom linéaire, Δf = {1e3 * r['f'][1]:.2f} mHz"]
    if horizontal:
        fig = make_subplots(rows=1, cols=2, subplot_titles=sous_titres, horizontal_spacing=0.08,
                            column_widths=[0.62, 0.38])
        pos = [(1, 1), (1, 2)]
    else:
        fig = make_subplots(rows=2, cols=1, subplot_titles=sous_titres, vertical_spacing=0.17,
                            row_heights=[0.6, 0.4])
        pos = [(1, 1), (2, 1)]
    lissage = fmt_num(reglages.get("lissage"), 0)
    ions = ions_choisis(reglages)
    tracer_psd_panneau(fig, r, *pos[0], log_x=True, legende=True, lissage=lissage, ions=ions)
    tracer_psd_panneau(fig, r, *pos[1], log_x=False, legende=False, lissage=lissage, ions=ions)

    # Pentes automatiques
    resume = []
    mode = reglages.get("ajustement") or "aucun"
    if mode != "aucun":
        fmin = fmt_num(reglages.get("fmin"), FIT_FMIN)
        fmax = fmt_num(reglages.get("fmax"), FIT_FMAX)
        for comp in composantes_choisies(reglages):
            nom = COMPOSANTES[comp][0]
            try:
                aj = ajustement_cache((cle, tuple(sel) if cible == "sel" else None, comp, fmin, fmax, mode),
                                      r["f"], r[comp], fmin, fmax, mode)
            except ValueError as exc:
                resume.append(html.Div(f"{nom} : {exc}", style={"color": "firebrick"}))
                continue
            tracer_ajustement(fig, aj, comp, *pos[0], legende=True)
            morceaux = [f"{s['pente']:.2f} ± {s['err']:.2f} ({s['f1']:.3g}–{s['f2']:.3g} Hz)"
                        for s in aj["segments"]]
            ruptures = ", ".join(f"{fb:.3g} Hz" for fb in aj["ruptures"]) or "aucune"
            resume.append(html.Div([html.B(f"{nom} : "), f"pentes {' | '.join(morceaux)} — ruptures : {ruptures}"
                                    + (" (nombre choisi par BIC)" if aj["auto"] else "")],
                                   style={"color": COMPOSANTES[comp][2]}))
    # Pentes manuelles
    mes_pentes = [p for p in (pentes or []) if p["fig"] == cible]
    mon_attente = attente if attente and attente.get("fig") == cible else None
    tracer_manuelles(fig, mes_pentes, mon_attente, *pos[0])
    if mes_pentes:
        resume.append(html.Div(
            [html.B("Manuelles : ")] + [f"m{i} = {pente_manuelle(p):.2f} ({p['f1']:.3g}→{p['f2']:.3g} Hz)   "
                                        for i, p in enumerate(mes_pentes, 1)],
            style={"color": COUL_MANUELLE}))

    fig.update_annotations(font_size=11, selector=dict(text=sous_titres[0]))
    fig.update_annotations(font_size=11, selector=dict(text=sous_titres[1]))
    fig.update_layout(title=dict(text=titre, font_size=13), height=height, template="plotly_white",
                      margin=dict(l=65, r=15, t=70, b=45), hovermode="closest",
                      uirevision=f"{cle}|{sel if cible == 'sel' else ''}",
                      legend=dict(orientation="h", x=0, y=-0.13 if horizontal else -0.08, font_size=11))
    return fig, resume


def disque_mercure(fig, row, col):
    """Mercure : jour (X > 0) clair, nuit sombre."""
    th = np.linspace(-np.pi / 2, np.pi / 2, 60)
    for signe, coul in ((1, "#d9a65a"), (-1, "#5a4630")):
        fig.add_trace(go.Scatter(x=np.r_[signe * np.cos(th), 0], y=np.r_[np.sin(th), 0], fill="toself",
                                 fillcolor=coul, line_width=0, mode="lines", hoverinfo="skip",
                                 showlegend=False), row=row, col=col)


def fig_orbite(cle, sel=None):
    """Orbite sur la fenêtre (X-Y, X-Z) et position en fonction du temps."""
    d = MEMOIRE[cle]
    xyz = d["xyz"]
    if xyz is None:
        return fig_vide("pas de données de position", 420)
    tp, P = xyz
    ok = np.isfinite(P).all(axis=1)
    tp, P = tp[ok], P[ok]
    if len(tp) < 2:
        return fig_vide("position insuffisante", 420)
    pas = max(1, int(np.ceil(len(tp) / MAX_POINTS_ORBITE)))
    garde = np.zeros(len(tp), bool)
    garde[::pas] = True
    garde[[0, -1]] = True
    tp, P = tp[garde], P[garde]
    x, y, z = P.T
    r = np.linalg.norm(P, axis=1)
    alt = (r - 1.0) * tr.R_MERCURE_KM
    lat = np.degrees(np.arcsin(np.clip(z / r, -1, 1)))
    lt = 12 + np.unwrap(np.arctan2(y, x)) * 12 / np.pi
    lt -= 24 * np.floor(np.median(lt) / 24)
    dans = (tp >= np.datetime64(d["deb"])) & (tp <= np.datetime64(d["fin"]))
    dans_sel = (tp >= np.datetime64(sel[0])) & (tp <= np.datetime64(sel[1])) if sel else np.zeros_like(dans)
    mp_x, mp_rho = tr.magnetopause()

    fig = make_subplots(rows=3, cols=3, column_widths=[0.34, 0.34, 0.32],
                        specs=[[{"rowspan": 3}, {"rowspan": 3}, {}], [None, None, {}], [None, None, {}]],
                        subplot_titles=["X-Y, vue du nord (Soleil à droite)", "X-Z, vue depuis l'aube (−Y)",
                                        "position de MESSENGER", "", ""],
                        horizontal_spacing=0.07, vertical_spacing=0.05)
    survol = "%{customdata}<br>(%{x:.3f}, %{y:.3f}) R_M<extra></extra>"
    temps_txt = np.datetime_as_string(tp, unit="s")
    morceaux = [(np.ones_like(dans), "trajectoire (fenêtre)", dict(color="gray", width=1.2)),
                (dans, "évènement", dict(color="red", width=3.5)),
                (dans_sel, "sélection", dict(color="purple", width=5))]
    for col, v, yl, mp in ((1, y, "Y_MSO [R_M]", [mp_rho, -mp_rho]),
                           (2, z, "Z_MSO [R_M]", [tr.Z_DIP + mp_rho, tr.Z_DIP - mp_rho])):
        disque_mercure(fig, 1, col)
        for i, cy in enumerate(mp):
            fig.add_trace(go.Scatter(x=mp_x, y=cy, mode="lines", line=dict(color="darkorange", width=1),
                                     name="magnétopause (Winslow+2013)", legendgroup="mp",
                                     showlegend=(col == 1 and i == 0), hoverinfo="skip"), row=1, col=col)
        for masque, nom, ligne in morceaux:
            if masque.any():
                fig.add_trace(go.Scatter(x=np.where(masque, x, np.nan), y=np.where(masque, v, np.nan),
                                         mode="lines", line=ligne, name=nom, legendgroup=nom,
                                         showlegend=col == 1, customdata=temps_txt, hovertemplate=survol),
                              row=1, col=col)
        fig.add_trace(go.Scatter(x=[x[0]], y=[v[0]], mode="markers", marker=dict(color="royalblue", size=9),
                                 name="début fenêtre", legendgroup="d", showlegend=col == 1,
                                 hoverinfo="skip"), row=1, col=col)
        fig.add_trace(go.Scatter(x=[x[-1]], y=[v[-1]], mode="markers",
                                 marker=dict(color="black", size=9, symbol="square"),
                                 name="fin fenêtre", legendgroup="f", showlegend=col == 1,
                                 hoverinfo="skip"), row=1, col=col)
        # sens de parcours, au milieu de la trajectoire (donc au centre de l'évènement)
        n = len(x)
        m = max(1, n // 10)
        a, b = max(0, n // 2 - m), min(n - 1, n // 2 + m)
        if b > a:
            xd, yd = ("x", "y") if col == 1 else ("x2", "y2")
            fig.add_annotation(x=x[b], y=v[b], ax=x[a], ay=v[a], xref=xd, yref=yd, axref=xd, ayref=yd,
                               showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=1.6, text="")
        u_all, v_all = np.r_[x, -1.1, 1.1], np.r_[v, -1.1, 1.1]
        mu, mv = 0.15 * np.ptp(u_all), 0.15 * np.ptp(v_all)
        fig.update_xaxes(range=[u_all.min() - mu, u_all.max() + mu], title_text="X_MSO [R_M]", row=1, col=col)
        fig.update_yaxes(range=[v_all.min() - mv, v_all.max() + mv], title_text=yl,
                         scaleanchor="x" if col == 1 else "x2", scaleratio=1, row=1, col=col)
    # rotation de Mercure dans la vue du nord : antihoraire (crépuscule en +Y, aube en −Y)
    for (x0, x1, yy, txt, haut) in ((0.3, -0.3, 1.15, "crépuscule (+Y)", True),
                                    (-0.3, 0.3, -1.15, "aube (−Y)", False)):
        fig.add_annotation(x=x1, y=yy, ax=x0, ay=yy, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2, arrowwidth=1.3, text="")
        fig.add_annotation(x=0, y=yy + (0.12 if haut else -0.12), xref="x", yref="y", text=txt,
                           showarrow=False, font=dict(size=10))

    # position en fonction du temps
    for v, lab, row in ((alt, "altitude [km]", 1), (lat, "latitude [°]", 2), (lt, "heure locale [h]", 3)):
        fig.add_trace(go.Scatter(x=tp, y=v, mode="lines", line=dict(color="dimgray", width=1.3), showlegend=False,
                                 hovertemplate="%{x|%H:%M:%S}<br>" + lab + " = %{y:.2f}<extra></extra>"),
                      row=row, col=3)
        fig.update_yaxes(title_text=lab, title_font_size=10, row=row, col=3)
        for a, b, coul in [(d["deb"], d["fin"], COUL_EVT)] + \
                          ([(sel[0], sel[1], COUL_SEL)] if sel else []):
            fig.add_vrect(x0=a, x1=b, fillcolor=coul, line_width=0, layer="below", row=row, col=3)
        fig.update_xaxes(tickformat="%H:%M", showticklabels=(row == 3), row=row, col=3)
        if row > 1:
            fig.update_xaxes(matches="x3", row=row, col=3)
    fig.add_hline(y=0, line=dict(color="black", dash="dot", width=0.8), row=2, col=3)
    fig.update_xaxes(title_text="UTC", row=3, col=3)
    fig.update_layout(height=540, template="plotly_white", uirevision=cle,
                      margin=dict(l=60, r=15, t=40, b=40),
                      legend=dict(orientation="h", x=0, y=-0.14, font_size=11))
    return fig


# ======================================================================
# Évènements étudiés (data/JSONS) : fichiers, catégories, arborescence
# ======================================================================
def nom_fichier(deb, fin):
    """Nom du JSON d'un évènement, tiré de ses dates : 2011-05-15_0926-0931.json (secondes si non nulles)."""
    def hm(d):
        return f"{d:%H%M}" + (f"{d:%S}" if d.second else "")
    fin_txt = hm(fin) if fin.date() == deb.date() else f"{fin:%Y-%m-%d}_{hm(fin)}"
    return f"{deb:%Y-%m-%d}_{hm(deb)}-{fin_txt}.json"


def chemin_json(nom):
    """Chemin d'un JSON de data/JSONS ; refuse tout nom qui sortirait du dossier."""
    p = DOSSIER_JSONS / Path(nom).name
    if p.suffix != ".json" or p.name.startswith("_"):
        raise ValueError(f"nom de fichier invalide : {nom}")
    return p


def lire_json(nom):
    return json.loads(chemin_json(nom).read_text(encoding="utf-8"))


def lister_jsons():
    """Résumé des JSON de data/JSONS : nom, dates, catégories, notes."""
    sortie = []
    if not DOSSIER_JSONS.is_dir():
        return sortie
    for p in sorted(DOSSIER_JSONS.glob("*.json")):
        if p.name.startswith("_"):
            continue
        try:
            c = json.loads(p.read_text(encoding="utf-8"))
            sortie.append(dict(nom=p.name, debut=str(c.get("debut", "")), fin=str(c.get("fin", "")),
                               categories=[str(x) for x in c.get("categories") or []],
                               notes=str(c.get("notes") or "")))
        except Exception:
            sortie.append(dict(nom=p.name, debut=p.stem, fin="", categories=["(illisible)"], notes=""))
    return sortie


def mtime_json(nom):
    """Date de modification d'un JSON (None s'il n'existe pas) : sert à détecter la sauvegarde d'un camarade."""
    try:
        return chemin_json(nom).stat().st_mtime
    except (OSError, ValueError):
        return None


def signature_jsons():
    """Change dès qu'un JSON est créé, modifié ou supprimé (par soi ou par un camarade)."""
    if not DOSSIER_JSONS.is_dir():
        return ""
    return "|".join(f"{p.name}:{p.stat().st_mtime_ns}" for p in sorted(DOSSIER_JSONS.glob("*.json")))


def etiquettes_avec_parents(cats):
    """« KH/flanc aube » -> {« KH », « KH/flanc aube »} : choisir « KH » montre aussi ses sous-étiquettes."""
    sortie = set()
    for c in cats:
        morceaux = c.split("/")
        sortie |= {"/".join(morceaux[:i]) for i in range(1, len(morceaux) + 1)}
    return sortie


def evenements_filtres(filtre, choisies=None):
    """Évènements de data/JSONS passant le filtre texte et ayant toutes les étiquettes choisies, par date."""
    evts = lister_jsons()
    f = (filtre or "").strip().lower()
    if f:
        evts = [e for e in evts if f in " ".join([e["nom"], e["debut"], e["notes"]] + e["categories"]).lower()]
    for c in choisies or []:
        evts = [e for e in evts if c in etiquettes_avec_parents(e["categories"])]
    return sorted(evts, key=lambda e: e["debut"])


def ligne_fichier(e, dossier, courant, retrait):
    return html.Div(
        [html.Span("📝" if e["notes"].strip() else "📄"), html.Span(etiquette_evt(e), className="nom")],
        id={"type": "fichier", "nom": e["nom"], "dossier": dossier}, n_clicks=0,
        className="ligne fichier" + (" actif" if e["nom"] == courant else ""),
        title=(" · ".join(e["categories"]) + ("\n\n" + e["notes"][:300] if e["notes"] else "")) or e["nom"],
        style={"paddingLeft": f"{retrait}px"})


def vue_etiquettes(filtre, choisies, courant):
    """Navigation par étiquettes : puces cliquables (cumulatives), puis les évènements qui les ont toutes."""
    choisies = list(choisies or [])
    tous = evenements_filtres(filtre)
    comptes = {}
    for e in tous:
        for c in etiquettes_avec_parents(e["categories"]):
            comptes[c] = comptes.get(c, 0) + 1
    for c in categories_connues():
        comptes.setdefault(c, 0)
    visibles = evenements_filtres(filtre, choisies)
    possibles = set().union(*(etiquettes_avec_parents(e["categories"]) for e in visibles)) if visibles else set()
    puces = [html.Span(f"{c} ({n})", id={"type": "puce", "cat": c}, n_clicks=0,
                       className="puce" + (" choisie" if c in choisies else "")
                       + ("" if c in choisies or c in possibles else " grise"),
                       style={"marginLeft": f"{8 * c.count('/')}px"})
             for c, n in sorted(comptes.items(), key=lambda kv: kv[0].lower())]
    entete = html.Div(
        ("Étiquettes choisies : " + " + ".join(choisies)) if choisies else "Cliquer des étiquettes pour filtrer",
        style={"fontSize": "11px", "color": "#666", "margin": "6px 0 4px"})
    lignes = [ligne_fichier(e, "#etiquettes", courant, 6) for e in visibles]
    if not lignes:
        lignes = [html.Div("Aucun évènement", style={"color": "#777", "fontSize": "13px", "padding": "6px"})]
    return [html.Div(puces, style={"display": "flex", "flexWrap": "wrap", "gap": "4px"}), entete,
            html.Div(f"{len(visibles)} évènement{'s' if len(visibles) > 1 else ''}",
                     style={"fontSize": "11px", "color": "#888", "borderTop": "1px solid #eee",
                            "paddingTop": "4px"})] + lignes


def categories_connues():
    cats = set()
    try:
        cats |= set(json.loads(FICHIER_CATEGORIES.read_text(encoding="utf-8")))
    except Exception:
        pass
    for e in lister_jsons():
        cats |= set(e["categories"])
    return sorted(c for c in cats if c)


def est_dans(cat, cible):
    """cat vaut cible ou en est une sous-catégorie (« KH/flanc aube » est dans « KH »)."""
    return cat == cible or cat.startswith(cible + "/")


def evenements_avec(cible):
    return [e for e in lister_jsons() if any(est_dans(c, cible) for c in e["categories"])]


def supprimer_categorie(cible):
    """
    Retire la catégorie cible (et ses sous-catégories) de _categories.json et de tous les JSON qui la portent.
    Renvoie les noms des JSON modifiés. Le reste de chaque JSON est réécrit tel quel.
    """
    modifies = []
    for e in evenements_avec(cible):
        p = chemin_json(e["nom"])
        c = json.loads(p.read_text(encoding="utf-8"))
        c["categories"] = [x for x in c.get("categories") or [] if not est_dans(str(x), cible)]
        p.write_text(json.dumps(c, indent=2, ensure_ascii=False), encoding="utf-8")
        modifies.append(e["nom"])
    restantes = [x for x in categories_connues() if not est_dans(x, cible)]
    DOSSIER_JSONS.mkdir(parents=True, exist_ok=True)
    FICHIER_CATEGORIES.write_text(json.dumps(restantes, indent=2, ensure_ascii=False), encoding="utf-8")
    return modifies


def retirer_json(nom):
    """Retire un évènement de la liste : son JSON part dans data/JSONS/_corbeille/ (horodaté), pas effacé."""
    src = chemin_json(nom)
    CORBEILLE.mkdir(parents=True, exist_ok=True)
    dest = CORBEILLE / f"{src.stem}__retire_{datetime.now():%Y%m%d-%H%M%S}.json"
    src.replace(dest)
    return dest


def enregistrer_categories(nouvelles):
    DOSSIER_JSONS.mkdir(parents=True, exist_ok=True)
    cats = sorted(set(categories_connues()) | {c for c in nouvelles if c})
    FICHIER_CATEGORIES.write_text(json.dumps(cats, indent=2, ensure_ascii=False), encoding="utf-8")
    return cats


def options_categories(en_plus=()):
    return [{"label": c, "value": c} for c in sorted(set(categories_connues()) | {c for c in en_plus if c})]


def nettoyer_categorie(c):
    """« KH / flanc aube » -> « KH/flanc aube » : les « / » créent des sous-dossiers dans l'explorateur."""
    return "/".join(x.strip() for x in str(c or "").split("/") if x.strip())


def etiquette_evt(e):
    deb, fin = e["debut"][:16], e["fin"]
    return f"{deb} → {fin[11:16] if fin[:10] == deb[:10] else fin[:16]}"


def arbre(filtre, fermes, courant):
    """Arborescence façon Finder : un dossier par catégorie (sous-dossiers via « / »), un fichier par évènement."""
    evts = lister_jsons()
    f = (filtre or "").strip().lower()
    if f:
        evts = [e for e in evts if f in " ".join([e["nom"], e["debut"], e["notes"]] + e["categories"]).lower()]
    racine = {"_": [], "sous": {}}
    for e in evts:
        for cat in (e["categories"] or ["Sans catégorie"]):
            noeud = racine
            for morceau in cat.split("/"):
                noeud = noeud["sous"].setdefault(morceau, {"_": [], "sous": {}})
            noeud["_"].append(e)
    if not f:  # catégories créées mais encore vides
        for cat in categories_connues():
            noeud = racine
            for morceau in cat.split("/"):
                noeud = noeud["sous"].setdefault(morceau, {"_": [], "sous": {}})
    fermes = set(fermes or [])

    def compte(noeud):
        return len({e["nom"] for e in noeud["_"]} | {n for s in noeud["sous"].values() for n in compte_noms(s)})

    def compte_noms(noeud):
        return {e["nom"] for e in noeud["_"]} | {n for s in noeud["sous"].values() for n in compte_noms(s)}

    def rendu(noeud, chemin, prof):
        lignes = []
        cles = sorted(noeud["sous"], key=lambda k: (k == "Sans catégorie", k.lower()))
        for nom in cles:
            sous, ch = noeud["sous"][nom], f"{chemin}/{nom}" if chemin else nom
            ouvert = ch not in fermes
            lignes.append(html.Div(
                [html.Span("▾" if ouvert else "▸", className="chevron"), html.Span("📂" if ouvert else "📁"),
                 html.Span(nom, className="nom"), html.Span(str(compte(sous)), className="compte")],
                id={"type": "dossier", "chemin": ch}, n_clicks=0, className="ligne dossier",
                style={"paddingLeft": f"{6 + 16 * prof}px"}))
            if ouvert:
                lignes += rendu(sous, ch, prof + 1)
        for e in sorted(noeud["_"], key=lambda e: e["debut"]):
            lignes.append(ligne_fichier(e, chemin, courant, 22 + 16 * prof))
        return lignes

    lignes = rendu(racine, "", 0)
    if not lignes:
        return html.Div("Aucun évènement sauvegardé : charger un évènement puis « Sauver dans le projet ».",
                        style={"color": "#777", "fontSize": "13px", "padding": "8px"})
    return lignes


# ======================================================================
# Interface
# ======================================================================
def options_evenements():
    """Menu des évènements de events.py ; ✓ devant ceux qui ont déjà un JSON dans data/JSONS."""
    opts = []
    for groupe, liste in tr.GROUPES.items():
        for i, e in enumerate(liste):
            try:
                fait = chemin_json(nom_fichier(parse_date(e.start), parse_date(e.end))).exists()
            except ValueError:
                fait = False
            opts.append({"label": f"{'✓ ' if fait else ''}{groupe} — {e.start} → {e.end[-5:]}",
                         "value": f"{groupe}|{i}"})
    return opts


def options_ions():
    return [{"label": f"{ion} (A={A}, q={q})", "value": ion} for ion, (A, q) in IONS_DISPO.items()]


STYLE_CHAMP = {"width": "190px", "padding": "4px", "fontFamily": "monospace"}
STYLE_PETIT = {"width": "70px", "padding": "4px"}
STYLE_BOUTON = {"padding": "6px 14px", "cursor": "pointer", "fontWeight": "bold"}
STYLE_BOUTON2 = {"padding": "5px 10px", "cursor": "pointer"}
STYLE_BLOC = {"display": "flex", "gap": "10px", "alignItems": "center", "flexWrap": "wrap"}
STYLE_CADRE = STYLE_BLOC | {"background": "#f5f5f7", "padding": "8px 12px", "borderRadius": "6px", "margin": "6px 0"}
STYLE_RESUME = {"fontSize": "13px", "fontFamily": "monospace", "margin": "2px 0 8px", "lineHeight": "1.6"}
STYLE_FACTEUR = {"background": "rgba(31,119,180,0.15)", "border": "1px solid rgba(31,119,180,0.6)",
                 "borderRadius": "5px", "padding": "3px 8px", "display": "flex", "gap": "6px",
                 "alignItems": "center"}

app = Dash(__name__, title="MESSENGER — KH explorer")
app.index_string = app.index_string.replace("{%css%}", """{%css%}
<style>
  .ligne { display: flex; align-items: center; gap: 6px; padding: 3px 6px; border-radius: 5px; cursor: pointer;
           font-size: 13px; user-select: none; white-space: nowrap; }
  .ligne:hover { background: #e8eef7; }
  .ligne .nom { overflow: hidden; text-overflow: ellipsis; }
  .ligne.dossier .nom { font-weight: 600; }
  .ligne .chevron { width: 10px; color: #666; font-size: 11px; }
  .ligne .compte { margin-left: auto; color: #888; font-size: 11px; background: #eee; border-radius: 8px;
                   padding: 0 6px; }
  .ligne.fichier { font-family: ui-monospace, monospace; font-size: 12px; }
  .ligne.actif { background: #cfe0fb; }
  .puce { font-size: 12px; padding: 2px 8px; border-radius: 10px; background: #eef1f5; cursor: pointer;
          user-select: none; border: 1px solid #d5dbe3; }
  .puce:hover { background: #dde6f3; }
  .puce.choisie { background: #2f6fd6; color: white; border-color: #2f6fd6; }
  .puce.grise { opacity: 0.4; }
</style>""")
STYLE_EXPLORATEUR = {"flexShrink": 0, "borderRight": "1px solid #ddd", "paddingRight": "8px",
                     "position": "sticky", "top": 0, "alignSelf": "flex-start", "maxHeight": "100vh",
                     "overflowY": "auto"}
EXPLORATEUR = html.Div(id="explorateur", style=STYLE_EXPLORATEUR | {"width": "290px"}, children=[
    html.Div(style={"display": "flex", "alignItems": "center", "gap": "4px"}, children=[
        html.Button("⟨", id="btn-replier", title="replier / déplier l'explorateur", style=STYLE_BOUTON2),
        html.Div(id="explorateur-titre", style={"flex": "1", "display": "flex", "alignItems": "center", "gap": "4px"},
                 children=[html.B("Évènements étudiés", style={"fontSize": "14px", "flex": "1"}),
                           html.Button("◀", id="btn-precedent", title="évènement précédent de la liste",
                                       style=STYLE_BOUTON2),
                           html.Button("▶", id="btn-suivant", title="évènement suivant de la liste",
                                       style=STYLE_BOUTON2),
                           html.Button("⟳", id="btn-rafraichir", title="relire data/JSONS", style=STYLE_BOUTON2)])]),
    html.Div(id="explorateur-contenu", children=[
        dcc.RadioItems(id="mode-arbre", value="dossiers", inline=True,
                       options=[{"label": " 📁 dossiers ", "value": "dossiers"},
                                {"label": " 🏷 étiquettes", "value": "etiquettes"}],
                       style={"fontSize": "13px", "margin": "6px 0 2px"}),
        dcc.Input(id="filtre-arbre", type="search", placeholder="filtrer (date, note, catégorie)…", debounce=False,
                  style={"width": "100%", "boxSizing": "border-box", "padding": "4px", "margin": "4px 0 6px"}),
        html.Div(id="arbre"),
        html.Div(id="info-sync", style={"fontSize": "10px", "color": "#999", "marginTop": "8px"}),
    ]),
])

NOTES = html.Div(style=STYLE_CADRE | {"alignItems": "flex-start"}, children=[
    html.Div(style={"flex": "2", "minWidth": "320px"}, children=[
        html.B("Notes sur l'évènement"),
        dcc.Textarea(id="notes", value="", placeholder="Observations, hypothèses, à vérifier…",
                     style={"width": "100%", "height": "78px", "marginTop": "4px", "fontFamily": "inherit"})]),
    html.Div(style={"flex": "1", "minWidth": "280px"}, children=[
        html.B("Catégories"),
        dcc.Dropdown(id="categories", options=[], value=[], multi=True, placeholder="aucune",
                     style={"marginTop": "4px"}),
        html.Div(style=STYLE_BLOC | {"marginTop": "6px"}, children=[
            dcc.Input(id="nouvelle-cat", type="text", placeholder="nouvelle catégorie (a/b = sous-dossier)",
                      debounce=False, style={"flex": "1", "padding": "4px"}),
            html.Button("＋ créer", id="btn-cat", style=STYLE_BOUTON2)]),
        html.Div(style=STYLE_BLOC | {"marginTop": "6px"}, children=[
            dcc.Dropdown(id="cat-a-supprimer", options=[], placeholder="catégorie désuète…", clearable=True,
                         style={"flex": "1", "minWidth": "160px"}),
            html.Button("🗑 supprimer", id="btn-suppr-cat", style=STYLE_BOUTON2),
            html.Button("confirmer", id="btn-suppr-confirmer",
                        style=STYLE_BOUTON2 | {"display": "none", "background": "#f8d0d0"})]),
        html.Div(id="statut-cat", style={"fontSize": "12px", "marginTop": "3px"})]),
])

def mise_en_page():
    """Recalculée à chaque ouverture de la page : menu events.py (✓) et catégories reflètent data/JSONS."""
    return html.Div(style={"fontFamily": "system-ui, sans-serif", "padding": "10px 14px", "display": "flex",
                                 "gap": "14px"}, children=[EXPLORATEUR, html.Div(style={"flex": "1", "minWidth": 0}, children=[
        html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"}, children=[
            html.H3("MESSENGER — exploration interactive (B, PSD, orbite)", style={"margin": "4px 0 10px"}),
            html.Div(style=STYLE_BLOC, children=[
                html.Span(id="statut-sauvegarde", style={"fontSize": "12px"}),
            html.Button("écraser quand même", id="btn-ecraser", style=STYLE_BOUTON2 | {"display": "none"}),
                html.Button("💾 Sauver dans le projet", id="btn-sauver-projet",
                            style=STYLE_BOUTON | {"background": "#d9f2dd"},
                            title="écrit data/JSONS/<dates>.json (remplace la version précédente de cet évènement)"),
                html.Button("🗑 retirer", id="btn-retirer", style=STYLE_BOUTON2,
                        title="retirer l'évènement ouvert de la liste (JSON déplacé dans data/JSONS/_corbeille)"),
            html.Button("confirmer le retrait", id="btn-retirer-confirmer",
                        style=STYLE_BOUTON2 | {"display": "none", "background": "#f8d0d0"}),
            html.Button("⬇ Exporter", id="btn-sauver", style=STYLE_BOUTON2, title="télécharger le JSON"),
                dcc.Upload(html.Button("📂 Importer", style=STYLE_BOUTON2), id="importer",
                           accept=".json,application/json"),
                dcc.Download(id="telechargement"),
            ]),
        ]),
        NOTES,

        html.Div(style=STYLE_BLOC, children=[
            dcc.Dropdown(id="choix-evt", options=options_evenements(), placeholder="Évènement de events.py…",
                         style={"width": "380px"}, clearable=True),
            html.Label("début"), dcc.Input(id="debut", type="text", value="2011-05-15 09:26", style=STYLE_CHAMP,
                                           debounce=True),
            html.Label("fin"), dcc.Input(id="fin", type="text", value="2011-05-15 09:31", style=STYLE_CHAMP,
                                         debounce=True),
            html.Div(style=STYLE_FACTEUR, children=[
                html.Label("facteur fenêtre"),
                dcc.Input(id="facteur", type="number", value=tr.FACTEUR_FENETRE, min=1, step=0.5, style=STYLE_PETIT)]),
            dcc.Checklist(id="force", options=[{"label": " retélécharger depuis AMDA", "value": "force"}], value=[]),
            html.Button("Charger", id="btn-charger", style=STYLE_BOUTON),
        ]),
        dcc.Loading(html.Div(id="statut", style={"margin": "8px 0", "fontSize": "14px"}), type="dot"),

        # Réglages communs aux deux PSD
        html.Div(style=STYLE_CADRE, children=[
            html.B("Réglages PSD"),
            html.Label("ions"),
            dcc.Dropdown(id="ions", options=options_ions(), value=list(tr.IONS), multi=True,
                         placeholder="aucune fréquence cyclotron", style={"minWidth": "330px"}),
            html.Label("lissage (décade)"),
            html.Div(dcc.Slider(id="lissage", min=0, max=0.5, step=0.01, value=0,
                                marks={0: "aucun", 0.1: "0,1", 0.2: "0,2", 0.3: "0,3", 0.4: "0,4", 0.5: "0,5"},
                                tooltip={"placement": "bottom"}), style={"width": "280px"}),
        ]),
        html.Div(style=STYLE_CADRE, children=[
            html.Label("pentes auto"),
            dcc.Dropdown(id="ajustement", value="auto", clearable=False, style={"width": "190px"},
                         options=[{"label": "aucune", "value": "aucun"},
                                  {"label": "ruptures : auto (BIC)", "value": "auto"},
                                  {"label": "1 rupture", "value": "1"},
                                  {"label": "2 ruptures", "value": "2"},
                                  {"label": "3 ruptures", "value": "3"}]),
            dcc.Dropdown(id="composante", value="perp", clearable=False, style={"width": "110px"},
                         options=[{"label": "P⊥", "value": "perp"}, {"label": "P∥", "value": "para"},
                                  {"label": "P⊥ et P∥", "value": "both"}]),
            html.Label("de"), dcc.Input(id="fmin", type="number", value=FIT_FMIN, min=0, step="any",
                                        debounce=True, style=STYLE_PETIT),
            html.Label("à"), dcc.Input(id="fmax", type="number", value=FIT_FMAX, min=0, step="any",
                                       debounce=True, style=STYLE_PETIT),
            html.Label("Hz"),
            html.Span("|", style={"color": "#aaa"}),
            dcc.Checklist(id="mode-manuel", options=[{"label": " pente manuelle (2 clics sur le log-log)",
                                                      "value": "on"}], value=[]),
            html.Button("annuler la dernière", id="btn-annuler", style=STYLE_BOUTON2),
            html.Button("effacer", id="btn-effacer", style=STYLE_BOUTON2),
        ]),

        html.Div(style={"display": "flex", "gap": "8px"}, children=[
            html.Div([dcc.Graph(id="fig-b", figure=fig_vide("Choisir un intervalle puis « Charger »", 620),
                                config=CONFIG_FIG),
                      html.Div(style=STYLE_BLOC, children=[
                          html.Button("↺ fenêtre initiale", id="btn-recentrer",
                                      style=STYLE_BOUTON | {"background": "rgba(31,119,180,0.15)"},
                                      title="revenir à la fenêtre ×facteur, bornes de B recalées sur cette fenêtre"),
                          dcc.Checklist(id="auto-y", value=["on"], style={"fontSize": "13px"},
                                        options=[{"label": " bornes de B = min / max de la zone affichée", "value": "on"}]),
                          html.Span("orange : évènement — violet : sélection",
                                    style={"fontSize": "12px", "color": "#555"})])],
                     style={"flex": "2.1", "minWidth": 0}),
            html.Div([dcc.Graph(id="fig-psd-fen", figure=fig_vide("PSD de la fenêtre", 620), config=CONFIG_FIG),
                      html.Div(id="resume-fen", style=STYLE_RESUME)],
                     style={"flex": "1", "minWidth": 0}),
        ]),

        html.H4("PSD d'une zone sélectionnée", style={"margin": "14px 0 6px"}),
        html.Div("Outil « Box Select » de la barre Plotly sur les courbes de B (calcul automatique), "
                 "bornes saisies à la main, zone actuellement affichée, ou évènement seul.",
                 style={"fontSize": "13px", "color": "#555", "marginBottom": "6px"}),
        html.Div(style=STYLE_BLOC, children=[
            html.Label("début"), dcc.Input(id="sel-debut", type="text", style=STYLE_CHAMP | {"width": "230px"}),
            html.Label("fin"), dcc.Input(id="sel-fin", type="text", style=STYLE_CHAMP | {"width": "230px"}),
            html.Button("PSD sur ces bornes", id="btn-sel", style=STYLE_BOUTON),
            html.Button("PSD de la zone affichée", id="btn-vue", style=STYLE_BOUTON),
            html.Button("PSD de l'évènement seul", id="btn-evt", style=STYLE_BOUTON | {"background": "#ffe2c6"}),
            html.Span(id="statut-sel", style={"fontSize": "13px"}),
        ]),
        dcc.Loading(dcc.Graph(id="fig-psd-sel", figure=fig_vide("Aucune zone sélectionnée", 500),
                              config=CONFIG_FIG), type="dot"),
        html.Div(id="resume-sel", style=STYLE_RESUME),

        html.H4("Orbite et position", style={"margin": "14px 0 6px"}),
        dcc.Graph(id="fig-orbite", figure=fig_vide("", 540), config=CONFIG_FIG),

        dcc.Store(id="cle"),
        dcc.Store(id="vue"),
        dcc.Store(id="selection"),
        dcc.Store(id="pentes-man", data=[]),
        dcc.Store(id="pente-attente"),
        dcc.Store(id="config-importee"),
        dcc.Store(id="config-brute"),
        dcc.Store(id="fichier-courant"),
        dcc.Store(id="dossiers-fermes", data=[]),
        dcc.Store(id="version-jsons", data=0),
    dcc.Store(id="etiquettes-choisies", data=[]),
    dcc.Store(id="explorateur-replie", data=False),
    dcc.Store(id="mtime-courant"),            # date du JSON ouvert : détecte la sauvegarde d'un camarade
    dcc.Store(id="signature-jsons"),
    dcc.Interval(id="sondage", interval=SONDAGE_S * 1000),
    ])])


app.layout = mise_en_page


@app.callback(Output("debut", "value"), Output("fin", "value"),
              Input("choix-evt", "value"), prevent_initial_call=True)
def choisir_evenement(val):
    if not val:
        return no_update, no_update
    groupe, i = val.split("|")
    e = tr.GROUPES[groupe][int(i)]
    return e.start, e.end


@app.callback(Output("cle", "data"), Output("statut", "children"), Output("fig-b", "figure"),
              Output("selection", "data", allow_duplicate=True), Output("vue", "data", allow_duplicate=True),
              Output("pentes-man", "data", allow_duplicate=True),
              Output("pente-attente", "data", allow_duplicate=True),
              Output("notes", "value", allow_duplicate=True), Output("categories", "value", allow_duplicate=True),
              Output("fichier-courant", "data", allow_duplicate=True),
              Output("categories", "options", allow_duplicate=True),
              Output("mtime-courant", "data", allow_duplicate=True),
              Input("btn-charger", "n_clicks"), Input("config-importee", "data"),
              State("debut", "value"), State("fin", "value"), State("facteur", "value"), State("force", "value"),
              State("choix-evt", "value"),
              prevent_initial_call=True)
def bouton_charger(_, config, debut, fin, facteur, force, choix_evt):
    depuis_config = ctx.triggered_id == "config-importee"
    if depuis_config:
        if not config:
            return (no_update,) * 12
        debut, fin = config.get("debut"), config.get("fin")
        facteur = config.get("facteur")
    try:
        deb, fin_ = parse_date(debut), parse_date(fin)
        if fin_ <= deb:
            raise ValueError("la fin doit être postérieure au début")
        facteur = max(1.0, fmt_num(facteur, tr.FACTEUR_FENETRE))
        cle = charger(deb, fin_, facteur, bool(force))
    except (ValueError, tr.DonneesInsuffisantes) as exc:
        return (no_update, html.Span(str(exc), style={"color": "firebrick"})) + (no_update,) * 10
    except Exception as exc:  # réseau, AMDA, speasy…
        return ((no_update, html.Span(f"{type(exc).__name__} : {exc}", style={"color": "firebrick"}))
                + (no_update,) * 10)
    d = MEMOIRE[cle]
    texte = f"Chargé : {fmt_date(d['wd'])} → {fmt_date(d['wf'])} — " + " ; ".join(d["notes"])
    if depuis_config:
        sel, vue, pentes = config.get("selection"), config.get("vue"), config.get("pentes_manuelles") or []
        attente = config.get("pente_en_cours")
        texte = "Configuration importée. " + texte
        notes, cats, courant, mtime = no_update, no_update, no_update, no_update   # posés à l'ouverture
    else:
        sel, vue, pentes, attente = None, None, [], None
        # Nouvel évènement : si un JSON existe déjà, on en reprend les notes et catégories (pas d'écrasement
        # involontaire à la prochaine sauvegarde) ; sinon catégorie « source/<liste> » si choisi dans events.py
        courant = nom_fichier(deb, fin_)
        try:
            ancien = lire_json(courant)
            notes, cats = ancien.get("notes") or "", ancien.get("categories") or []
            texte += f" — notes et catégories reprises de data/JSONS/{courant}"
            mtime = mtime_json(courant)
        except (OSError, ValueError):
            notes, courant, mtime = "", None, None
            cats = [f"source/{choix_evt.split('|')[0]}"] if choix_evt else []
    # Dash retire d'un menu les valeurs absentes de ses options : on envoie les deux ensemble
    options = no_update if cats is no_update else options_categories(cats)
    return (cle, html.Span(texte, style={"color": "darkgreen"}), fig_b(cle, vue, sel), sel, vue, pentes, attente,
            notes, cats, courant, options, mtime)


@app.callback(Output("selection", "data"), Output("statut-sel", "children"),
              Output("pentes-man", "data", allow_duplicate=True),
              Output("pente-attente", "data", allow_duplicate=True),
              Input("fig-b", "selectedData"), Input("btn-sel", "n_clicks"), Input("btn-vue", "n_clicks"),
              Input("btn-evt", "n_clicks"),
              State("sel-debut", "value"), State("sel-fin", "value"), State("vue", "data"), State("cle", "data"),
              State("pentes-man", "data"), State("pente-attente", "data"),
              prevent_initial_call=True)
def definir_selection(selected, _b1, _b2, _b3, s_deb, s_fin, vue, cle, pentes, attente):
    if not cle or cle not in MEMOIRE:
        return no_update, "charger d'abord des données", no_update, no_update
    d = MEMOIRE[cle]
    try:
        if ctx.triggered_id == "fig-b":
            if not selected or not selected.get("range"):
                return (no_update,) * 4
            # la zone vient du sous-graphe où la boîte a été tracée (clé x, x2, x3 ou x4)
            xr = next(v for k, v in selected["range"].items() if k.startswith("x"))
            a, b = parse_date(xr[0]), parse_date(xr[1])
        elif ctx.triggered_id == "btn-vue":
            a, b = (parse_date(vue[0]), parse_date(vue[1])) if vue else (d["wd"], d["wf"])
            a, b = max(a, d["wd"]), min(b, d["wf"])
        elif ctx.triggered_id == "btn-evt":
            a, b = d["deb"], d["fin"]
        else:
            a, b = parse_date(s_deb), parse_date(s_fin)
    except (ValueError, StopIteration) as exc:
        return no_update, html.Span(str(exc), style={"color": "firebrick"}), no_update, no_update
    if b < a:
        a, b = b, a
    # Les pentes manuelles de l'ancienne sélection n'ont plus de sens sur la nouvelle PSD
    pentes = [p for p in (pentes or []) if p["fig"] != "sel"]
    attente = None if attente and attente.get("fig") == "sel" else attente
    return [fmt_date(a, ms=True), fmt_date(b, ms=True)], "", pentes, attente


@app.callback(Output("sel-debut", "value"), Output("sel-fin", "value"),
              Input("selection", "data"), prevent_initial_call=True)
def afficher_bornes(sel):
    return (sel[0], sel[1]) if sel else ("", "")


@app.callback(Output("vue", "data"), Input("fig-b", "relayoutData"), State("cle", "data"),
              prevent_initial_call=True)
def suivre_vue(relayout, cle):
    """Mémorise la plage de temps affichée (zoom / pan), quel que soit le sous-graphe manipulé."""
    if not relayout:
        return no_update
    for k, v in relayout.items():
        if re.fullmatch(r"xaxis\d*\.autorange", k) and v:
            d = MEMOIRE.get(cle)
            return [fmt_date(d["wd"]), fmt_date(d["wf"])] if d else None
        if re.fullmatch(r"xaxis\d*\.range\[0\]", k):
            return [str(v), str(relayout[k.replace("[0]", "[1]")])]
        if re.fullmatch(r"xaxis\d*\.range", k):
            return [str(v[0]), str(v[1])]
    return no_update


@app.callback(Output("fig-b", "figure", allow_duplicate=True), Output("vue", "data", allow_duplicate=True),
              Input("btn-recentrer", "n_clicks"), State("cle", "data"), prevent_initial_call=True)
def recadrer(n_r, cle):
    """Ramène la vue de B sur la fenêtre initiale, avec des bornes de B recalées sur ses données."""
    if not cle or cle not in MEMOIRE:
        return no_update, no_update
    d = MEMOIRE[cle]
    a, b = d["wd"], d["wf"]
    patch = Patch()
    # Nouveau uirevision : sinon Plotly garde le zoom de l'utilisateur quand la plage demandée est égale
    # à celle de la figure d'origine (cas de « ↺ fenêtre »)
    patch["layout"]["uirevision"] = f"{cle}|{n_r or 0}"
    for ax in ("xaxis", "xaxis2", "xaxis3", "xaxis4"):
        patch["layout"][ax]["range"] = [fmt_date(a), fmt_date(b)]
        patch["layout"][ax]["autorange"] = False
    for ax, plage in zip(("yaxis", "yaxis2", "yaxis3", "yaxis4"), plages_b(d, a, b)):
        if plage:
            patch["layout"][ax]["range"] = plage
            patch["layout"][ax]["autorange"] = False
    return patch, [fmt_date(a), fmt_date(b)]


@app.callback(Output("fig-b", "figure", allow_duplicate=True),
              Input("vue", "data"), State("cle", "data"), State("auto-y", "value"), State("fig-b", "relayoutData"),
              prevent_initial_call=True)
def raffiner_b(vue, cle, auto_y, relayout):
    """
    Pleine résolution dans la zone affichée, min-max ailleurs : n'envoie que les nouvelles données.
    Si la case est cochée, recale aussi les bornes de chaque panneau de B sur le min / max de la zone affichée,
    sauf quand l'utilisateur vient lui-même de choisir une plage en B (zoom rectangle).
    """
    if not cle or cle not in MEMOIRE:
        return no_update
    d = MEMOIRE[cle]
    patch = Patch()
    for i, (x, y) in enumerate(donnees_affichees(d, vue)):
        patch["data"][i]["x"] = x
        patch["data"][i]["y"] = y
    zoom_y_utilisateur = any(re.fullmatch(r"yaxis\d*\.range(\[0\])?", k) for k in (relayout or {}))
    if "on" in (auto_y or []) and not zoom_y_utilisateur:
        for ax, plage in zip(("yaxis", "yaxis2", "yaxis3", "yaxis4"), plages_b(d, *bornes_vue(d, vue))):
            if plage:
                patch["layout"][ax]["range"] = plage
                patch["layout"][ax]["autorange"] = False
    return patch


@app.callback(Output("fig-b", "figure", allow_duplicate=True),
              Input("selection", "data"), State("cle", "data"), prevent_initial_call=True)
def surligner_selection(sel, cle):
    if not cle or cle not in MEMOIRE:
        return no_update
    patch = Patch()
    s = (parse_date(sel[0]), parse_date(sel[1])) if sel else None
    patch["layout"]["shapes"] = formes_b(MEMOIRE[cle], s)
    return patch


ENTREES_REGLAGES = [Input("ions", "value"), Input("lissage", "value"), Input("ajustement", "value"),
                    Input("composante", "value"), Input("fmin", "value"), Input("fmax", "value"),
                    Input("pentes-man", "data"), Input("pente-attente", "data")]


def reglages_dict(ions, lissage, ajustement, composante, fmin, fmax):
    return dict(ions=ions, lissage=lissage, ajustement=ajustement, composante=composante, fmin=fmin, fmax=fmax)


@app.callback(Output("fig-psd-fen", "figure"), Output("resume-fen", "children"),
              Input("cle", "data"), *ENTREES_REGLAGES, prevent_initial_call=True)
def psd_fenetre(cle, ions, lissage, ajustement, composante, fmin, fmax, pentes, attente):
    return construire_psd(cle, None, "fen", reglages_dict(ions, lissage, ajustement, composante, fmin, fmax),
                          pentes, attente)


@app.callback(Output("fig-psd-sel", "figure"), Output("resume-sel", "children"),
              Input("selection", "data"), *ENTREES_REGLAGES, State("cle", "data"), prevent_initial_call=True)
def psd_selection(sel, ions, lissage, ajustement, composante, fmin, fmax, pentes, attente, cle):
    return construire_psd(cle, sel, "sel", reglages_dict(ions, lissage, ajustement, composante, fmin, fmax),
                          pentes, attente)


@app.callback(Output("pentes-man", "data"), Output("pente-attente", "data"),
              Input("fig-psd-fen", "clickData"), Input("fig-psd-sel", "clickData"),
              Input("btn-annuler", "n_clicks"), Input("btn-effacer", "n_clicks"),
              State("mode-manuel", "value"), State("pentes-man", "data"), State("pente-attente", "data"),
              prevent_initial_call=True)
def pentes_manuelles(clic_fen, clic_sel, _a, _e, mode, pentes, attente):
    pentes = list(pentes or [])
    if ctx.triggered_id == "btn-effacer":
        return [], None
    if ctx.triggered_id == "btn-annuler":
        if attente:
            return no_update, None
        return pentes[:-1], no_update
    if "on" not in (mode or []):
        return no_update, no_update
    cible = "fen" if ctx.triggered_id == "fig-psd-fen" else "sel"
    clic = clic_fen if cible == "fen" else clic_sel
    if not clic or not clic.get("points"):
        return no_update, no_update
    pt = clic["points"][0]
    if pt.get("customdata") != "log" or not pt.get("x") or not pt.get("y") or pt["x"] <= 0 or pt["y"] <= 0:
        return no_update, no_update
    point = dict(fig=cible, f=float(pt["x"]), p=float(pt["y"]))
    if not attente or attente.get("fig") != cible:
        return no_update, point
    if point["f"] == attente["f"]:
        return no_update, no_update
    (f1, p1), (f2, p2) = sorted([(attente["f"], attente["p"]), (point["f"], point["p"])])
    return pentes + [dict(fig=cible, f1=f1, p1=p1, f2=f2, p2=p2)], None


@app.callback(Output("fig-orbite", "figure"), Input("cle", "data"), Input("selection", "data"),
              prevent_initial_call=True)
def orbite(cle, sel):
    if not cle or cle not in MEMOIRE:
        return no_update
    s = (parse_date(sel[0]), parse_date(sel[1])) if sel else None
    return fig_orbite(cle, s)


# ======================================================================
# Sauvegarde / import de la configuration
# ======================================================================
def arrondi(v, n=6):
    """Nombre à n chiffres significatifs pour un JSON lisible."""
    return float(f"{float(v):.{n}g}")


def resultats_psd(cle, sel, cible, reglages, pentes):
    """Tout ce qu'affiche une figure de PSD, en clair : caractéristiques, raies cyclotron, pentes auto et manuelles."""
    if not cle or cle not in MEMOIRE or (cible == "sel" and not sel):
        return None
    d = MEMOIRE[cle]
    r = psd(cle, sel if cible == "sel" else None)
    sortie = dict(titre=titre_psd(d, sel, cible),
                  intervalle=list(sel) if cible == "sel" else [fmt_date(d["wd"]), fmt_date(d["wf"])])
    if isinstance(r, str):
        sortie["erreur"] = r
        return sortie
    sortie.update(N=int(r["N"]), fs_Hz=arrondi(r["fs"]), delta_f_Hz=arrondi(r["f"][1]),
                  B_moy_nT=arrondi(r["B_moy"]), B_std_nT=arrondi(r["B_std"]),
                  frequences_cyclotron_Hz={ion: arrondi(tr.cyclo(IONS_DISPO[ion][0], r["B_moy"], IONS_DISPO[ion][1]))
                                           for ion in ions_choisis(reglages)})
    mode = reglages.get("ajustement") or "aucun"
    auto = {}
    if mode != "aucun":
        fmin = fmt_num(reglages.get("fmin"), FIT_FMIN)
        fmax = fmt_num(reglages.get("fmax"), FIT_FMAX)
        for comp in composantes_choisies(reglages):
            nom = COMPOSANTES[comp][0]
            try:
                aj = ajustement_cache((cle, tuple(sel) if cible == "sel" else None, comp, fmin, fmax, mode),
                                      r["f"], r[comp], fmin, fmax, mode)
            except ValueError as exc:
                auto[nom] = dict(erreur=str(exc))
                continue
            auto[nom] = dict(
                plage_Hz=[fmin, fmax], mode=mode, nb_ruptures=aj["k"],
                choix_nb_ruptures="BIC" if aj["auto"] else "imposé",
                ruptures_Hz=[arrondi(fb) for fb in aj["ruptures"]],
                segments=[dict(f_debut_Hz=arrondi(sg["f1"]), f_fin_Hz=arrondi(sg["f2"]),
                               pente=arrondi(sg["pente"], 4), ecart_type=arrondi(sg["err"], 3))
                          for sg in aj["segments"]])
    sortie["pentes_auto"] = auto
    sortie["pentes_manuelles"] = [
        dict(nom=f"m{i}", f1_Hz=arrondi(p["f1"]), psd1=arrondi(p["p1"]), f2_Hz=arrondi(p["f2"]),
             psd2=arrondi(p["p2"]), pente=arrondi(pente_manuelle(p), 4))
        for i, p in enumerate([p for p in (pentes or []) if p["fig"] == cible], 1)]
    return sortie


ETATS_CONFIG = [State("cle", "data"), State("debut", "value"), State("fin", "value"), State("facteur", "value"),
                State("choix-evt", "value"), State("force", "value"),
                State("selection", "data"), State("vue", "data"), State("ions", "value"), State("lissage", "value"),
                State("ajustement", "value"), State("composante", "value"), State("fmin", "value"),
                State("fmax", "value"), State("mode-manuel", "value"), State("pentes-man", "data"),
                State("pente-attente", "data"), State("auto-y", "value"), State("notes", "value"),
                State("categories", "value")]


def construire_config(cle, debut, fin, facteur, choix_evt, force, sel, vue, ions, lissage, ajustement,
                      composante, fmin, fmax, mode_manuel, pentes, attente, auto_y, notes, categories):
    """
    Configuration JSON complète :
      - réglages (relus à l'ouverture) : intervalle, zoom, sélection, réglages PSD, pentes manuelles,
        notes et catégories ;
      - "resultats" : tout ce qui est affiché à l'écran, en clair (caractéristiques des PSD, fréquences
        cyclotron, pentes automatiques avec ruptures et écarts-types, pentes manuelles avec leur valeur).
        Cette partie sert de trace ; à l'ouverture, les résultats sont recalculés à partir des réglages.
    """
    reglages = reglages_dict(ions, lissage, ajustement, composante, fmin, fmax)
    charge = cle in MEMOIRE
    if charge:  # intervalle réellement affiché, même si les champs ont été modifiés depuis
        d = MEMOIRE[cle]
        debut, fin, facteur = fmt_date(d["deb"]), fmt_date(d["fin"]), d["facteur"]
    evt = None
    if choix_evt:
        groupe, i = choix_evt.split("|")
        e = tr.GROUPES[groupe][int(i)]
        evt = dict(groupe=groupe, indice=int(i), debut=e.start, fin=e.end)
    return dict(
        version=VERSION_CONFIG, application="webapp.py (KH-EA-3A)",
        sauvegarde=f"{datetime.now():%Y-%m-%d %H:%M:%S}",
        evenement_events_py=evt,
        debut=debut, fin=fin, facteur=facteur,
        notes=notes or "", categories=[c for c in (categories or []) if c],
        fenetre=[fmt_date(d["wd"]), fmt_date(d["wf"])] if charge else None,
        retelecharger_amda=bool(force),
        vue=vue, echelle_B_auto="on" in (auto_y or []), selection=sel,
        psd=reglages | dict(mode_pente_manuelle="on" in (mode_manuel or [])),
        pentes_manuelles=pentes or [], pente_en_cours=attente,
        resultats=dict(
            notes_donnees=MEMOIRE[cle]["notes"] if charge else None,
            psd_fenetre=resultats_psd(cle, sel, "fen", reglages, pentes),
            psd_selection=resultats_psd(cle, sel, "sel", reglages, pentes)),
    )


@app.callback(Output("telechargement", "data"), Input("btn-sauver", "n_clicks"), *ETATS_CONFIG,
              prevent_initial_call=True)
def exporter(_, *etats):
    """« Exporter » : télécharge le JSON dans le navigateur."""
    config = construire_config(*etats)
    try:
        nom = nom_fichier(parse_date(config["debut"]), parse_date(config["fin"]))
    except ValueError:
        nom = "config_webapp.json"
    return dict(content=json.dumps(config, indent=2, ensure_ascii=False), filename=nom)


@app.callback(Output("statut-sauvegarde", "children"), Output("version-jsons", "data"),
              Output("fichier-courant", "data"), Output("choix-evt", "options"),
              Output("mtime-courant", "data"), Output("btn-ecraser", "style"),
              Input("btn-sauver-projet", "n_clicks"), Input("btn-ecraser", "n_clicks"), *ETATS_CONFIG,
              State("version-jsons", "data"), State("mtime-courant", "data"),
              prevent_initial_call=True)
def sauver_projet(_s, _e, *etats):
    """
    « Sauver dans le projet » : écrit (ou remplace) data/JSONS/<dates>.json.
    Si le fichier a changé sur le disque depuis qu'on l'a ouvert (sauvegarde d'un camarade), ou s'il existe déjà
    alors qu'on ne l'a pas ouvert, on n'écrase pas sans confirmation (« écraser quand même »).
    """
    *etats, version, mtime_ouvert = etats
    cache = STYLE_BOUTON2 | {"display": "none"}
    config = construire_config(*etats)
    try:
        nom = nom_fichier(parse_date(config["debut"]), parse_date(config["fin"]))
        actuel = mtime_json(nom)
        if ctx.triggered_id != "btn-ecraser" and actuel is not None and actuel != mtime_ouvert:
            quand = datetime.fromtimestamp(actuel)
            qui = "modifié" if mtime_ouvert is not None else "déjà présent"
            msg = (f"⚠ data/JSONS/{nom} {qui} sur le disque ({quand:%d/%m %H:%M:%S}), sans doute par un camarade : "
                   "rouvrir l'évènement depuis l'explorateur pour voir sa version, ou")
            return (html.Span(msg, style={"color": "#b35c00"}), no_update, no_update, no_update, no_update,
                    STYLE_BOUTON2 | {"background": "#ffe2c6"})
        DOSSIER_JSONS.mkdir(parents=True, exist_ok=True)
        chemin_json(nom).write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        enregistrer_categories(config["categories"])
    except (ValueError, OSError) as exc:
        return (html.Span(f"Sauvegarde impossible : {exc}", style={"color": "firebrick"}), no_update, no_update,
                no_update, no_update, cache)
    return (html.Span(f"✓ data/JSONS/{nom} ({datetime.now():%H:%M:%S})", style={"color": "darkgreen"}),
            (version or 0) + 1, nom, options_evenements(), mtime_json(nom), cache)


@app.callback(Output("categories", "options"), Output("categories", "value", allow_duplicate=True),
              Output("nouvelle-cat", "value"), Output("version-jsons", "data", allow_duplicate=True),
              Input("btn-cat", "n_clicks"), Input("nouvelle-cat", "n_submit"), Input("version-jsons", "data"),
              Input("categories", "value"), State("nouvelle-cat", "value"),
              prevent_initial_call="initial_duplicate")
def gerer_categories(_b, _s, version, actuelles, nouvelle):
    """
    Crée une catégorie à la volée (et l'ajoute à l'évènement courant) ; tient la liste à jour, y compris avec
    les catégories de l'évènement ouvert (sinon le menu ne les afficherait pas).
    """
    actuelles = list(actuelles or [])
    if ctx.triggered_id in ("btn-cat", "nouvelle-cat"):
        cat = nettoyer_categorie(nouvelle)
        if not cat:
            return no_update, no_update, no_update, no_update
        cats = enregistrer_categories([cat])
        if cat not in actuelles:
            actuelles.append(cat)
        return options_categories(cats + actuelles), actuelles, "", (version or 0) + 1
    return options_categories(actuelles), no_update, no_update, no_update


@app.callback(Output("statut-sauvegarde", "children", allow_duplicate=True),
              Output("btn-retirer-confirmer", "style"),
              Input("btn-retirer", "n_clicks"), Input("fichier-courant", "data"), prevent_initial_call=True)
def preparer_retrait(_, courant):
    """1er clic sur « 🗑 retirer » : demande confirmation pour l'évènement ouvert."""
    cache = STYLE_BOUTON2 | {"display": "none", "background": "#f8d0d0"}
    if ctx.triggered_id != "btn-retirer":
        return no_update, cache   # autre évènement ouvert : on annule une demande en cours
    if not courant or mtime_json(courant) is None:
        return html.Span("Aucun évènement sauvegardé n'est ouvert.", style={"color": "#b35c00"}), cache
    return (html.Span(f"Retirer data/JSONS/{courant} de la liste ?", style={"color": "#b35c00"}),
            cache | {"display": "inline-block"})


@app.callback(Output("statut-sauvegarde", "children", allow_duplicate=True),
              Output("btn-retirer-confirmer", "style", allow_duplicate=True),
              Output("fichier-courant", "data", allow_duplicate=True),
              Output("mtime-courant", "data", allow_duplicate=True),
              Output("version-jsons", "data", allow_duplicate=True),
              Output("choix-evt", "options", allow_duplicate=True),
              Input("btn-retirer-confirmer", "n_clicks"), State("fichier-courant", "data"),
              State("version-jsons", "data"), prevent_initial_call=True)
def confirmer_retrait(_, courant, version):
    """2e clic : le JSON part dans la corbeille ; l'écran reste tel quel (on peut encore le re-sauver)."""
    cache = STYLE_BOUTON2 | {"display": "none", "background": "#f8d0d0"}
    if not courant:
        return no_update, cache, no_update, no_update, no_update, no_update
    try:
        dest = retirer_json(courant)
    except (OSError, ValueError) as exc:
        return (html.Span(f"Retrait impossible : {exc}", style={"color": "firebrick"}), cache,
                no_update, no_update, no_update, no_update)
    msg = f"✓ {courant} retiré (récupérable : data/JSONS/_corbeille/{dest.name})"
    return html.Span(msg, style={"color": "darkgreen"}), cache, None, None, (version or 0) + 1, options_evenements()


@app.callback(Output("cat-a-supprimer", "options"), Input("categories", "options"))
def options_suppression(_):
    """Toutes les catégories, y compris les dossiers parents (supprimer « KH » retire aussi « KH/… »)."""
    return [{"label": c, "value": c} for c in sorted(etiquettes_avec_parents(categories_connues()), key=str.lower)]


@app.callback(Output("statut-cat", "children"), Output("btn-suppr-confirmer", "style"),
              Input("btn-suppr-cat", "n_clicks"), Input("cat-a-supprimer", "value"), prevent_initial_call=True)
def preparer_suppression(_, cible):
    """1er clic : annonce ce qui sera modifié et montre « confirmer »."""
    cache = STYLE_BOUTON2 | {"display": "none", "background": "#f8d0d0"}
    if ctx.triggered_id != "btn-suppr-cat" or not cible:
        # menu vidé après une suppression : on laisse le message « ✓ supprimée » affiché
        return ("" if cible else no_update), cache
    evts = evenements_avec(cible)
    sous = sorted(c for c in etiquettes_avec_parents(categories_connues()) if est_dans(c, cible) and c != cible)
    msg = f"« {cible} »" + (f" et ses sous-catégories ({', '.join(sous)})" if sous else "")
    msg += (f" {'seront retirées' if sous else 'sera retirée'} de {len(evts)} évènement{'s' if len(evts) > 1 else ''} : "
            + ", ".join(etiquette_evt(e) for e in evts[:4]) + (" …" if len(evts) > 4 else "")) if evts \
        else " n'est utilisée par aucun évènement"
    return html.Span(msg + ".", style={"color": "#b35c00"}), cache | {"display": "inline-block"}


@app.callback(Output("statut-cat", "children", allow_duplicate=True),
              Output("btn-suppr-confirmer", "style", allow_duplicate=True),
              Output("cat-a-supprimer", "value"),
              Output("categories", "options", allow_duplicate=True),
              Output("categories", "value", allow_duplicate=True),
              Output("mtime-courant", "data", allow_duplicate=True),
              Output("version-jsons", "data", allow_duplicate=True),
              Output("etiquettes-choisies", "data", allow_duplicate=True),
              Input("btn-suppr-confirmer", "n_clicks"),
              State("cat-a-supprimer", "value"), State("categories", "value"), State("fichier-courant", "data"),
              State("version-jsons", "data"), State("etiquettes-choisies", "data"), prevent_initial_call=True)
def confirmer_suppression(_, cible, actuelles, courant, version, choisies):
    """2e clic : retire la catégorie partout ; l'évènement ouvert suit sans fausse alerte de conflit."""
    cache = STYLE_BOUTON2 | {"display": "none", "background": "#f8d0d0"}
    if not cible:
        return "", cache, no_update, no_update, no_update, no_update, no_update, no_update
    try:
        modifies = supprimer_categorie(cible)
    except (OSError, ValueError) as exc:
        return (html.Span(f"Suppression impossible : {exc}", style={"color": "firebrick"}), cache) + (no_update,) * 6
    restantes = [c for c in (actuelles or []) if not est_dans(c, cible)]
    mtime = mtime_json(courant) if courant in modifies else no_update
    msg = f"✓ « {cible} » supprimée" + (f", retirée de {len(modifies)} JSON" if modifies else "")
    return (html.Span(msg, style={"color": "darkgreen"}), cache, None, options_categories(restantes), restantes,
            mtime, (version or 0) + 1, [c for c in (choisies or []) if not est_dans(c, cible)])


@app.callback(Output("arbre", "children"),
              Input("btn-rafraichir", "n_clicks"), Input("filtre-arbre", "value"), Input("dossiers-fermes", "data"),
              Input("fichier-courant", "data"), Input("version-jsons", "data"), Input("signature-jsons", "data"),
              Input("mode-arbre", "value"), Input("etiquettes-choisies", "data"))
def afficher_arbre(_r, filtre, fermes, courant, _v, _sig, mode, choisies):
    if mode == "etiquettes":
        return vue_etiquettes(filtre, choisies, courant)
    return arbre(filtre, fermes, courant)


@app.callback(Output("signature-jsons", "data"), Output("info-sync", "children"),
              Input("sondage", "n_intervals"), State("signature-jsons", "data"))
def sonder_jsons(_, ancienne):
    """Relit data/JSONS toutes les SONDAGE_S s : l'explorateur montre les sauvegardes d'un camarade."""
    sig = signature_jsons()
    info = f"data/JSONS relu à {datetime.now():%H:%M:%S}"
    return (no_update if sig == ancienne else sig), info


@app.callback(Output("etiquettes-choisies", "data"), Input({"type": "puce", "cat": ALL}, "n_clicks"),
              State("etiquettes-choisies", "data"), prevent_initial_call=True)
def choisir_etiquette(_, choisies):
    if not ctx.triggered_id or not any(t["value"] for t in ctx.triggered):
        return no_update
    c = ctx.triggered_id["cat"]
    choisies = list(choisies or [])
    return [x for x in choisies if x != c] if c in choisies else choisies + [c]


@app.callback(Output("explorateur", "style"), Output("explorateur-contenu", "style"),
              Output("explorateur-titre", "style"), Output("btn-replier", "children"),
              Output("explorateur-replie", "data"),
              Input("btn-replier", "n_clicks"), State("explorateur-replie", "data"), prevent_initial_call=True)
def replier(_, replie):
    replie = not replie
    titre = {"flex": "1", "display": "flex", "alignItems": "center", "gap": "4px"}
    if replie:
        return (STYLE_EXPLORATEUR | {"width": "34px", "overflowY": "hidden"}, {"display": "none"},
                titre | {"display": "none"}, "⟩", True)
    return STYLE_EXPLORATEUR | {"width": "290px"}, {}, titre, "⟨", False


# Plotly ne suit que les redimensionnements de la fenêtre : après repli / dépli, on en simule un pour que les
# graphiques occupent la place libérée
app.clientside_callback(
    """function(replie) { setTimeout(() => window.dispatchEvent(new Event('resize')), 50); return ''; }""",
    Output("explorateur-titre", "title"), Input("explorateur-replie", "data"), prevent_initial_call=True)


@app.callback(Output("config-brute", "data", allow_duplicate=True),
              Output("fichier-courant", "data", allow_duplicate=True),
              Output("mtime-courant", "data", allow_duplicate=True),
              Input("btn-precedent", "n_clicks"), Input("btn-suivant", "n_clicks"),
              State("mode-arbre", "value"), State("filtre-arbre", "value"), State("etiquettes-choisies", "data"),
              State("fichier-courant", "data"), prevent_initial_call=True)
def naviguer(_p, _s, mode, filtre, choisies, courant):
    """◀ / ▶ : évènement précédent / suivant (par date) parmi ceux que montre l'explorateur."""
    evts = evenements_filtres(filtre, choisies if mode == "etiquettes" else None)
    if not evts:
        return no_update, no_update, no_update
    noms = [e["nom"] for e in evts]
    pas = -1 if ctx.triggered_id == "btn-precedent" else 1
    if courant in noms:
        i = noms.index(courant) + pas
        if not 0 <= i < len(noms):
            return no_update, no_update, no_update
    else:
        i = 0 if pas > 0 else len(noms) - 1
    return ouvrir(noms[i])


@app.callback(Output("dossiers-fermes", "data"), Input({"type": "dossier", "chemin": ALL}, "n_clicks"),
              State("dossiers-fermes", "data"), prevent_initial_call=True)
def ouvrir_fermer(clics, fermes):
    if not ctx.triggered_id or not any(t["value"] for t in ctx.triggered):
        return no_update   # arbre redessiné, pas un clic
    ch = ctx.triggered_id["chemin"]
    fermes = set(fermes or [])
    fermes ^= {ch}
    return sorted(fermes)


def ouvrir(nom):
    """JSON de data/JSONS -> (config à appliquer, fichier courant, date de modification)."""
    try:
        config = lire_json(nom)
    except (OSError, ValueError) as exc:
        return {"_erreur": f"Ouverture de {nom} impossible : {exc}"}, no_update, no_update
    config["_source"] = f"data/JSONS/{nom}"
    return config, nom, mtime_json(nom)


@app.callback(Output("config-brute", "data", allow_duplicate=True),
              Output("fichier-courant", "data", allow_duplicate=True),
              Output("mtime-courant", "data", allow_duplicate=True),
              Input({"type": "fichier", "nom": ALL, "dossier": ALL}, "n_clicks"), prevent_initial_call=True)
def ouvrir_fichier(clics):
    """Clic sur un évènement de l'explorateur : ouvre son JSON."""
    if not ctx.triggered_id or not any(t["value"] for t in ctx.triggered):
        return no_update, no_update, no_update
    return ouvrir(ctx.triggered_id["nom"])


@app.callback(Output("config-brute", "data"), Output("importer", "contents"),
              Input("importer", "contents"), State("importer", "filename"), prevent_initial_call=True)
def importer(contenu, nom_fichier_):
    """« Importer » : JSON choisi sur l'ordinateur."""
    if not contenu:
        return no_update, no_update
    try:
        config = json.loads(base64.b64decode(contenu.split(",", 1)[1]).decode("utf-8"))
        if not isinstance(config, dict):
            raise ValueError("ce n'est pas une configuration de webapp.py")
    except Exception as exc:
        return {"_erreur": f"Import de {nom_fichier_} impossible : {exc}"}, None
    config["_source"] = nom_fichier_
    return config, None


@app.callback(Output("debut", "value", allow_duplicate=True), Output("fin", "value", allow_duplicate=True),
              Output("facteur", "value"), Output("ions", "value"),
              Output("lissage", "value"), Output("ajustement", "value"),
              Output("composante", "value"), Output("fmin", "value"), Output("fmax", "value"),
              Output("mode-manuel", "value"), Output("force", "value"), Output("auto-y", "value"),
              Output("notes", "value"), Output("categories", "value"), Output("choix-evt", "value"),
              Output("config-importee", "data"), Output("statut", "children", allow_duplicate=True),
              Output("categories", "options", allow_duplicate=True),
              Input("config-brute", "data"), prevent_initial_call=True)
def appliquer_config(config):
    """Remet l'écran dans l'état d'un JSON (explorateur ou import), puis déclenche le chargement des données."""
    if not config:
        return (no_update,) * 18
    try:
        if "_erreur" in config:
            raise ValueError(config["_erreur"])
        if "debut" not in config or "fin" not in config:
            raise ValueError("ce n'est pas une configuration de webapp.py")
        parse_date(config["debut"]), parse_date(config["fin"])
    except ValueError as exc:
        return (no_update,) * 16 + (html.Span(str(exc), style={"color": "firebrick"}), no_update)
    p = config.get("psd") or {}
    config["_import"] = datetime.now().isoformat()  # force le rechargement même pour le même fichier
    ions = p.get("ions")
    return (config["debut"], config["fin"], config.get("facteur", tr.FACTEUR_FENETRE),
            list(tr.IONS) if ions is None else [i for i in ions if i in IONS_DISPO],
            p.get("lissage", 0), p.get("ajustement", "auto"), p.get("composante", "perp"),
            p.get("fmin", FIT_FMIN), p.get("fmax", FIT_FMAX),
            ["on"] if p.get("mode_pente_manuelle") else [], ["force"] if config.get("retelecharger_amda") else [],
            [] if config.get("echelle_B_auto") is False else ["on"],
            config.get("notes") or "", config.get("categories") or [], None,
            config, f"Ouverture de {config.get('_source', 'la configuration')}…",
            options_categories(config.get("categories") or []))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Webapp d'exploration des données MESSENGER")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--host", default="127.0.0.1",
                    help="0.0.0.0 pour ouvrir l'application aux autres machines (réseau local, Tailscale)")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    print(f"Ouvrir http://127.0.0.1:{args.port}" + ("  (accessible aux autres machines)" if args.host != "127.0.0.1" else ""))
    if args.host != "127.0.0.1" and args.debug:
        ap.error("--debug est interdit avec --host : le débogueur permettrait d'exécuter du code à distance")
    app.run(debug=args.debug, port=args.port, host=args.host)
