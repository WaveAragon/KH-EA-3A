#!/usr/bin/env python3
"""
webapp.py : exploration interactive des données MESSENGER (AMDA via speasy), dans le navigateur.

Reprend la sortie de traitement.py (B, PSD, orbite, position) en figures Plotly zoomables, et permet de
sélectionner une zone temporelle sur les courbes de B pour en calculer la PSD.

Tous les calculs viennent de traitement.py (téléchargement + cache data/cache/, comblement des trous,
calcul_psd selon Gershman et al. 2015, fréquences cyclotron, magnétopause de Winslow) : ce fichier ne fait
que l'interface. Une modification de traitement.py se répercute donc ici.

Lancement (depuis la racine du dépôt) :
  pip install dash            (en plus de numpy, scipy, matplotlib, speasy)
  python webapp.py            puis ouvrir http://127.0.0.1:8050
  python webapp.py --port 8051 --debug

Utilisation :
  1. Choisir un évènement de events.py ou saisir début / fin ("YYYY-MM-DD HH:MM[:SS]"), puis « Charger ».
     Comme dans traitement.py, la fenêtre affichée vaut FACTEUR x la durée, centrée sur l'évènement.
  2. Molette / glisser pour zoomer, outil « Pan » de la barre Plotly pour se déplacer, double-clic pour revenir.
  3. Outil « Box Select » (barre Plotly) sur les courbes de B : la PSD de la zone sélectionnée se calcule
     automatiquement. On peut aussi saisir les bornes à la main, ou prendre la zone actuellement affichée.
"""
import argparse
import re
from collections import OrderedDict
from datetime import datetime

import numpy as np
from scipy.stats import chi2

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update, Patch

import traitement as tr

# ======================================================================
# Données en mémoire (application locale mono-utilisateur)
# ======================================================================
MEMOIRE = OrderedDict()   # clé -> dict(t, B, deb, fin, wd, wf, xyz, notes)
MEMOIRE_MAX = 6           # nombre de fenêtres gardées en mémoire
MAX_POINTS_AFFICHES = 300_000  # au-delà, B est décimé pour l'affichage seulement (jamais pour la PSD)

COUL_EVT = "rgba(255,127,14,0.15)"   # orange de traitement.py
COUL_SEL = "rgba(128,0,128,0.13)"    # violet : zone sélectionnée
CONFIG_FIG = {"scrollZoom": True, "displaylogo": False,
              "toImageButtonOptions": {"format": "png", "scale": 2}}


def parse_date(s):
    """Accepte "YYYY-MM-DD HH:MM", "YYYY-MM-DD HH:MM:SS[.ffffff]" et le séparateur T (format Plotly inclus)."""
    s = (s or "").strip().replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise ValueError(f"date illisible : « {s} » (format attendu YYYY-MM-DD HH:MM[:SS])")


def fmt_date(d):
    return f"{d:%Y-%m-%d %H:%M:%S}"


def charger(deb, fin, facteur, force):
    """Télécharge (ou relit le cache) et prépare B et la position. Renvoie la clé en mémoire."""
    marge = (fin - deb) * ((facteur - 1) / 2)
    wd, wf = (deb - marge).replace(microsecond=0), (fin + marge).replace(microsecond=0)
    if wd < tr.PLAGE_B[0] or wf > tr.PLAGE_B[1]:
        raise tr.DonneesInsuffisantes(
            f"fenêtre {fmt_date(wd)} → {fmt_date(wf)} hors de la plage de {tr.PRODUIT_B} "
            f"({tr.PLAGE_B[0]:%Y-%m-%d} → {tr.PLAGE_B[1]:%Y-%m-%d})")

    donnees_b = tr.telecharger(tr.PRODUIT_B, wd, wf, force)
    if donnees_b is None:
        raise tr.DonneesInsuffisantes("aucune donnée de champ magnétique AMDA sur cette fenêtre")
    t, B = donnees_b
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

    cle = f"{wd:%Y%m%dT%H%M%S}_{wf:%Y%m%dT%H%M%S}_{datetime.now():%H%M%S%f}"
    MEMOIRE[cle] = dict(t=t, B=B, deb=deb, fin=fin, wd=wd, wf=wf, xyz=xyz, notes=notes)
    while len(MEMOIRE) > MEMOIRE_MAX:
        MEMOIRE.popitem(last=False)
    return cle


# ======================================================================
# Figures
# ======================================================================
def vrects(deb, fin, couleur, lignes, nom):
    """Bandes verticales (une par sous-graphe) ; lignes = liste des (xref, yref domain)."""
    return [dict(type="rect", xref=xr, yref=f"{yr} domain", x0=deb, x1=fin, y0=0, y1=1,
                 fillcolor=couleur, line_width=0, layer="below", name=nom) for xr, yr in lignes]


LIGNES_B = [("x", "y"), ("x2", "y2"), ("x3", "y3"), ("x4", "y4")]


def formes_b(d, sel=None):
    """Bandes évènement / sélection et lignes B = 0 des quatre sous-graphes de B."""
    formes = [dict(type="line", xref=f"{xr} domain", yref=yr, x0=0, x1=1, y0=0, y1=0,
                   line=dict(color="black", dash="dot", width=0.8)) for xr, yr in LIGNES_B[:3]]
    formes += vrects(d["deb"], d["fin"], COUL_EVT, LIGNES_B, "evenement")
    if sel:
        formes += vrects(sel[0], sel[1], COUL_SEL, LIGNES_B, "selection")
    return formes


def fig_b(d):
    t, B = d["t"], d["B"]
    pas = max(1, int(np.ceil(len(t) / MAX_POINTS_AFFICHES)))
    t_aff, B_aff = t[::pas], B[::pas]
    B_abs = np.linalg.norm(B_aff, axis=1)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.025)
    series = [(B_aff[:, 0], "Bx"), (B_aff[:, 1], "By"), (B_aff[:, 2], "Bz"), (B_abs, "|B|")]
    for i, (y, nom) in enumerate(series, 1):
        # Marqueurs invisibles : nécessaires pour que l'outil de sélection Plotly renvoie la zone choisie
        fig.add_trace(go.Scattergl(x=t_aff, y=y, mode="lines+markers", name=nom, showlegend=False,
                                   line=dict(color="black", width=1),
                                   marker=dict(size=3, opacity=0),
                                   selected=dict(marker=dict(opacity=0)),
                                   unselected=dict(marker=dict(opacity=0)),
                                   hovertemplate="%{x|%H:%M:%S.%L}<br>" + nom + " = %{y:.2f} nT<extra></extra>"),
                      row=i, col=1)
        fig.update_yaxes(title_text=f"{nom} (nT)", row=i, col=1)
    duree = (d["fin"] - d["deb"]).total_seconds() / 60
    titre = (f"{d['deb']:%Y-%m-%d %H:%M:%S} → {d['fin']:%H:%M:%S} UTC (durée {duree:.1f} min) — "
             f"fenêtre {d['wd']:%H:%M:%S} → {d['wf']:%H:%M:%S}")
    if pas > 1:
        titre += f" — affichage décimé 1/{pas}"
    fig.update_layout(shapes=formes_b(d), title=dict(text=titre, font_size=13),
                      height=620, margin=dict(l=60, r=15, t=45, b=40),
                      dragmode="zoom", selectdirection="h", hovermode="x",
                      uirevision="b", template="plotly_white")
    fig.update_xaxes(range=[d["wd"], d["wf"]], showspikes=True, spikemode="across", spikethickness=1)
    fig.update_xaxes(title_text=f"UTC ({d['wd']:%Y-%m-%d})", tickformat="%H:%M:%S", row=4, col=1)
    return fig


def fig_vide(texte, height=300):
    fig = go.Figure()
    fig.add_annotation(text=texte, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font=dict(color="firebrick", size=13))
    fig.update_layout(height=height, template="plotly_white", xaxis_visible=False, yaxis_visible=False,
                      margin=dict(l=20, r=20, t=30, b=20))
    return fig


def tracer_psd_panneau(fig, r, row, col, log_x, legende, nom_axe_x):
    """Un panneau de PSD : log-log (log_x=True) ou zoom en fréquence linéaire jusqu'à ZOOM_LIN."""
    f, perp, para = r["f"], r["perp"], r["para"]
    m = slice(1, None) if log_x else (f > 0) & (f <= tr.ZOOM_LIN)
    mode = "lines" if log_x else "lines+markers"
    for y, nom, coul in ((perp, "P⊥", "purple"), (para, "P∥", "mediumseagreen")):
        fig.add_trace(go.Scatter(x=f[m], y=y[m], mode=mode, name=nom, legendgroup=nom, showlegend=legende,
                                 line=dict(color=coul, width=1.4), marker=dict(size=4),
                                 hovertemplate="f = %{x:.4g} Hz<br>" + nom + " = %{y:.3g} nT²/Hz<extra></extra>"),
                      row=row, col=col)

    f_min, f_max = (f[1], f[-1]) if log_x else (0, tr.ZOOM_LIN)
    xd = fig.get_subplot(row, col).xaxis.plotly_name.replace("axis", "")   # "x", "x2", ...
    yd = fig.get_subplot(row, col).yaxis.plotly_name.replace("axis", "")
    formes = []
    if log_x:  # bins atténués par le detrending
        formes.append(dict(type="rect", xref=xd, yref=f"{yd} domain", x0=f[1], x1=2 * f[1], y0=0, y1=1,
                           fillcolor="rgba(0,0,0,0.08)", line_width=0, layer="below"))
    for ion, (A, q) in tr.IONS.items():
        fc = tr.cyclo(A, r["B_moy"], q)
        if not f_min < fc < f_max:
            continue
        coul = tr.COULEURS_IONS[ion]
        formes.append(dict(type="rect", xref=xd, yref=f"{yd} domain",
                           x0=tr.cyclo(A, r["B_moy"] - r["B_std"], q), x1=tr.cyclo(A, r["B_moy"] + r["B_std"], q),
                           y0=0, y1=1, fillcolor=coul, opacity=0.15, line_width=0, layer="below"))
        formes.append(dict(type="line", xref=xd, yref=f"{yd} domain", x0=fc, x1=fc, y0=0, y1=1,
                           line=dict(color=coul, dash="dash", width=1.2)))
        # Les annotations sur un axe log se placent en log10(x)
        fig.add_annotation(x=np.log10(fc) if log_x else fc, y=1, xref=xd, yref=f"{yd} domain",
                           text=f"{ion} : {fc:.3g} Hz", textangle=-90, showarrow=False,
                           xanchor="right", yanchor="top", font=dict(size=10, color=coul))
    for s in formes:
        fig.add_shape(**s)

    if log_x:
        # Barre d'IC à 95 % (lissage sur 3 bins, fenêtre rectangulaire : 6 ddl), comme traitement.py
        nu = 6
        ic_bas, ic_haut = nu / chi2.ppf(0.975, nu), nu / chi2.ppf(0.025, nu)
        y_ok = np.r_[perp[1:], para[1:]]
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
    fig.update_xaxes(title_text=nom_axe_x, row=row, col=col)


def fig_psd(r, titre, horizontal, height):
    sous_titres = [f"⟨|B|⟩ = {r['B_moy']:.1f} ± {r['B_std']:.1f} nT, fs = {r['fs']:.1f} Hz, N = {r['N']}",
                   f"zoom linéaire, Δf = {1e3 * r['f'][1]:.2f} mHz"]
    if horizontal:
        fig = make_subplots(rows=1, cols=2, subplot_titles=sous_titres, horizontal_spacing=0.08)
        pos = [(1, 1), (1, 2)]
    else:
        fig = make_subplots(rows=2, cols=1, subplot_titles=sous_titres, vertical_spacing=0.17)
        pos = [(1, 1), (2, 1)]
    tracer_psd_panneau(fig, r, *pos[0], log_x=True, legende=True, nom_axe_x="Fréquence [Hz]")
    tracer_psd_panneau(fig, r, *pos[1], log_x=False, legende=False, nom_axe_x="Fréquence [Hz]")
    fig.update_annotations(font_size=11, selector=dict(text=sous_titres[0]))
    fig.update_annotations(font_size=11, selector=dict(text=sous_titres[1]))
    fig.update_layout(title=dict(text=titre, font_size=13), height=height, template="plotly_white",
                      margin=dict(l=65, r=15, t=70, b=45), hovermode="closest",
                      legend=dict(orientation="h", x=0, y=-0.12 if horizontal else -0.08))
    return fig


def psd_ou_message(t, B, titre, horizontal, height):
    try:
        r = tr.calcul_psd(t, B)
    except tr.DonneesInsuffisantes as exc:
        return fig_vide(f"{titre}<br>{exc}", height), str(exc)
    return fig_psd(r, titre, horizontal, height), ""


def disque_mercure(fig, row, col):
    """Mercure : jour (X > 0) clair, nuit sombre."""
    th = np.linspace(-np.pi / 2, np.pi / 2, 60)
    for signe, coul in ((1, "#d9a65a"), (-1, "#5a4630")):
        fig.add_trace(go.Scatter(x=np.r_[signe * np.cos(th), 0], y=np.r_[np.sin(th), 0], fill="toself",
                                 fillcolor=coul, line_width=0, mode="lines", hoverinfo="skip",
                                 showlegend=False), row=row, col=col)


def fig_orbite(d, sel=None):
    xyz = d["xyz"]
    if xyz is None:
        return fig_vide("pas de données de position", 420)
    tp, P = xyz
    ok = np.isfinite(P).all(axis=1)
    tp, P = tp[ok], P[ok]
    if len(tp) < 2:
        return fig_vide("position insuffisante", 420)
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
                        horizontal_spacing=0.07, vertical_spacing=0.05, shared_xaxes=False)
    survol = "%{customdata}<br>(%{x:.3f}, %{y:.3f}) R_M<extra></extra>"
    temps_txt = np.datetime_as_string(tp, unit="s")
    for col, v, yl, mp in ((1, y, "Y_MSO [R_M]", [mp_rho, -mp_rho]),
                           (2, z, "Z_MSO [R_M]", [tr.Z_DIP + mp_rho, tr.Z_DIP - mp_rho])):
        disque_mercure(fig, 1, col)
        for i, cy in enumerate(mp):
            fig.add_trace(go.Scatter(x=mp_x, y=cy, mode="lines", line=dict(color="darkorange", width=1),
                                     name="magnétopause (Winslow+2013)", legendgroup="mp",
                                     showlegend=(col == 1 and i == 0), hoverinfo="skip"), row=1, col=col)
        fig.add_trace(go.Scatter(x=x, y=v, mode="lines", line=dict(color="gray", width=1.2),
                                 name="trajectoire (fenêtre)", legendgroup="traj", showlegend=col == 1,
                                 customdata=temps_txt, hovertemplate=survol), row=1, col=col)
        if dans.any():
            fig.add_trace(go.Scatter(x=x[dans], y=v[dans], mode="lines", line=dict(color="red", width=3.5),
                                     name="évènement", legendgroup="evt", showlegend=col == 1,
                                     customdata=temps_txt[dans], hovertemplate=survol), row=1, col=col)
        if dans_sel.any():
            fig.add_trace(go.Scatter(x=x[dans_sel], y=v[dans_sel], mode="lines",
                                     line=dict(color="purple", width=5), name="sélection", legendgroup="sel",
                                     showlegend=col == 1, customdata=temps_txt[dans_sel],
                                     hovertemplate=survol), row=1, col=col)
        fig.add_trace(go.Scatter(x=[x[0]], y=[v[0]], mode="markers", marker=dict(color="royalblue", size=9),
                                 name="début fenêtre", legendgroup="d", showlegend=col == 1,
                                 hoverinfo="skip"), row=1, col=col)
        fig.add_trace(go.Scatter(x=[x[-1]], y=[v[-1]], mode="markers",
                                 marker=dict(color="black", size=9, symbol="square"),
                                 name="fin fenêtre", legendgroup="f", showlegend=col == 1,
                                 hoverinfo="skip"), row=1, col=col)
        # sens de parcours au milieu de la trajectoire
        n = len(x)
        m = max(1, n // 10)
        a, b = max(0, n // 2 - m), min(n - 1, n // 2 + m)
        xd = "x" if col == 1 else "x2"
        yd = "y" if col == 1 else "y2"
        fig.add_annotation(x=x[b], y=v[b], ax=x[a], ay=v[a], xref=xd, yref=yd, axref=xd, ayref=yd,
                           showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=1.6, text="")
        u_all, v_all = np.r_[x, -1.1, 1.1], np.r_[v, -1.1, 1.1]
        mu, mv = 0.15 * np.ptp(u_all), 0.15 * np.ptp(v_all)
        fig.update_xaxes(range=[u_all.min() - mu, u_all.max() + mu], title_text="X_MSO [R_M]", row=1, col=col)
        fig.update_yaxes(range=[v_all.min() - mv, v_all.max() + mv], title_text=yl,
                         scaleanchor=xd, scaleratio=1, row=1, col=col)
    # rotation de Mercure dans la vue du nord : antihoraire (crépuscule en +Y, aube en −Y)
    for (x0, x1, yy, txt, pos) in ((0.3, -0.3, 1.15, "crépuscule (+Y)", "top"),
                                   (-0.3, 0.3, -1.15, "aube (−Y)", "bottom")):
        fig.add_annotation(x=x1, y=yy, ax=x0, ay=yy, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2, arrowwidth=1.3, text="")
        fig.add_annotation(x=0, y=yy + (0.12 if pos == "top" else -0.12), xref="x", yref="y", text=txt,
                           showarrow=False, font=dict(size=10))

    # position en fonction du temps
    lignes_pos = [(alt, "altitude [km]", 1), (lat, "latitude [°]", 2), (lt, "heure locale [h]", 3)]
    for v, lab, row in lignes_pos:
        fig.add_trace(go.Scatter(x=tp, y=v, mode="lines", line=dict(color="dimgray", width=1.3),
                                 showlegend=False, hovertemplate="%{x|%H:%M:%S}<br>" + lab + " = %{y:.2f}<extra></extra>"),
                      row=row, col=3)
        fig.update_yaxes(title_text=lab, title_font_size=10, row=row, col=3)
        fig.add_vrect(x0=d["deb"], x1=d["fin"], fillcolor=COUL_EVT, line_width=0, layer="below", row=row, col=3)
        if sel:
            fig.add_vrect(x0=sel[0], x1=sel[1], fillcolor=COUL_SEL, line_width=0, layer="below", row=row, col=3)
        fig.update_xaxes(tickformat="%H:%M", showticklabels=(row == 3), row=row, col=3)
        if row > 1:
            fig.update_xaxes(matches="x3", row=row, col=3)
    fig.add_hline(y=0, line=dict(color="black", dash="dot", width=0.8), row=2, col=3)
    fig.update_xaxes(title_text="UTC", row=3, col=3)
    fig.update_layout(height=520, template="plotly_white", uirevision="orbite",
                      margin=dict(l=60, r=15, t=40, b=40),
                      legend=dict(orientation="h", x=0, y=-0.14, font_size=11))
    return fig


# ======================================================================
# Interface
# ======================================================================
def options_evenements():
    opts = []
    for groupe, liste in tr.GROUPES.items():
        for i, e in enumerate(liste):
            opts.append({"label": f"{groupe} — {e.start} → {e.end[-5:]}", "value": f"{groupe}|{i}"})
    return opts


STYLE_CHAMP = {"width": "190px", "padding": "4px", "fontFamily": "monospace"}
STYLE_BOUTON = {"padding": "6px 14px", "cursor": "pointer", "fontWeight": "bold"}
STYLE_BLOC = {"display": "flex", "gap": "10px", "alignItems": "center", "flexWrap": "wrap"}

app = Dash(__name__, title="MESSENGER — KH explorer")
app.layout = html.Div(style={"fontFamily": "system-ui, sans-serif", "padding": "10px 18px"}, children=[
    html.H3("MESSENGER — exploration interactive (B, PSD, orbite)", style={"margin": "4px 0 10px"}),

    html.Div(style=STYLE_BLOC, children=[
        dcc.Dropdown(id="choix-evt", options=options_evenements(), placeholder="Évènement de events.py…",
                     style={"width": "380px"}, clearable=True),
        html.Label("début"), dcc.Input(id="debut", type="text", value="2011-05-15 09:26", style=STYLE_CHAMP,
                                       debounce=True),
        html.Label("fin"), dcc.Input(id="fin", type="text", value="2011-05-15 09:31", style=STYLE_CHAMP,
                                     debounce=True),
        html.Label("facteur fenêtre"), dcc.Input(id="facteur", type="number", value=tr.FACTEUR_FENETRE, min=1,
                                                 step=0.5, style={"width": "70px", "padding": "4px"}),
        dcc.Checklist(id="force", options=[{"label": " retélécharger depuis AMDA", "value": "force"}], value=[]),
        html.Button("Charger", id="btn-charger", style=STYLE_BOUTON),
    ]),
    dcc.Loading(html.Div(id="statut", style={"margin": "8px 0", "fontSize": "14px"}), type="dot"),

    html.Div(style={"display": "flex", "gap": "8px"}, children=[
        html.Div(dcc.Graph(id="fig-b", figure=fig_vide("Choisir un intervalle puis « Charger »", 620),
                           config=CONFIG_FIG), style={"flex": "2.1", "minWidth": 0}),
        html.Div(dcc.Graph(id="fig-psd-fen", figure=fig_vide("PSD de la fenêtre", 620), config=CONFIG_FIG),
                 style={"flex": "1", "minWidth": 0}),
    ]),

    html.H4("PSD d'une zone sélectionnée", style={"margin": "14px 0 6px"}),
    html.Div("Outil « Box Select » de la barre Plotly sur les courbes de B (calcul automatique), "
             "ou bornes saisies à la main, ou zone actuellement affichée.",
             style={"fontSize": "13px", "color": "#555", "marginBottom": "6px"}),
    html.Div(style=STYLE_BLOC, children=[
        html.Label("début"), dcc.Input(id="sel-debut", type="text", style=STYLE_CHAMP | {"width": "230px"}),
        html.Label("fin"), dcc.Input(id="sel-fin", type="text", style=STYLE_CHAMP | {"width": "230px"}),
        html.Button("PSD sur ces bornes", id="btn-sel", style=STYLE_BOUTON),
        html.Button("PSD de la zone affichée", id="btn-vue", style=STYLE_BOUTON),
        html.Span(id="statut-sel", style={"fontSize": "13px"}),
    ]),
    dcc.Loading(dcc.Graph(id="fig-psd-sel", figure=fig_vide("Aucune zone sélectionnée", 480),
                          config=CONFIG_FIG), type="dot"),

    html.H4("Orbite et position", style={"margin": "14px 0 6px"}),
    dcc.Graph(id="fig-orbite", figure=fig_vide("", 520), config=CONFIG_FIG),

    dcc.Store(id="cle"),
    dcc.Store(id="vue"),
    dcc.Store(id="selection"),
])


@app.callback(Output("debut", "value"), Output("fin", "value"),
              Input("choix-evt", "value"), prevent_initial_call=True)
def choisir_evenement(val):
    if not val:
        return no_update, no_update
    groupe, i = val.split("|")
    e = tr.GROUPES[groupe][int(i)]
    return e.start, e.end


@app.callback(Output("cle", "data"), Output("statut", "children"), Output("fig-b", "figure"),
              Output("fig-psd-fen", "figure"), Output("selection", "data", allow_duplicate=True),
              Input("btn-charger", "n_clicks"),
              State("debut", "value"), State("fin", "value"), State("facteur", "value"), State("force", "value"),
              prevent_initial_call=True)
def bouton_charger(_, debut, fin, facteur, force):
    try:
        deb, fin_ = parse_date(debut), parse_date(fin)
        if fin_ <= deb:
            raise ValueError("la fin doit être postérieure au début")
        facteur = max(1.0, float(facteur or tr.FACTEUR_FENETRE))
        cle = charger(deb, fin_, facteur, bool(force))
    except (ValueError, tr.DonneesInsuffisantes) as exc:
        return no_update, html.Span(str(exc), style={"color": "firebrick"}), no_update, no_update, no_update
    except Exception as exc:  # réseau, AMDA, speasy…
        return (no_update, html.Span(f"{type(exc).__name__} : {exc}", style={"color": "firebrick"}),
                no_update, no_update, no_update)
    d = MEMOIRE[cle]
    fig_p, err = psd_ou_message(d["t"], d["B"], "PSD de la fenêtre affichée", horizontal=False, height=620)
    notes = d["notes"] + ([err] if err else [])
    statut = html.Span(f"Chargé : {fmt_date(d['wd'])} → {fmt_date(d['wf'])} — " + " ; ".join(notes),
                       style={"color": "darkgreen"})
    return cle, statut, fig_b(d), fig_p, None


@app.callback(Output("selection", "data"), Output("sel-debut", "value"), Output("sel-fin", "value"),
              Output("statut-sel", "children"),
              Input("fig-b", "selectedData"), Input("btn-sel", "n_clicks"), Input("btn-vue", "n_clicks"),
              State("sel-debut", "value"), State("sel-fin", "value"), State("vue", "data"), State("cle", "data"),
              prevent_initial_call=True)
def definir_selection(selected, _b1, _b2, s_deb, s_fin, vue, cle):
    if not cle or cle not in MEMOIRE:
        return no_update, no_update, no_update, "charger d'abord des données"
    try:
        if ctx.triggered_id == "fig-b":
            if not selected or not selected.get("range"):
                return no_update, no_update, no_update, no_update
            # la zone vient du sous-graphe où la boîte a été tracée (clé x, x2, x3 ou x4)
            xr = next(v for k, v in selected["range"].items() if k.startswith("x"))
            a, b = parse_date(xr[0]), parse_date(xr[1])
        elif ctx.triggered_id == "btn-vue":
            d = MEMOIRE[cle]
            a, b = (parse_date(vue[0]), parse_date(vue[1])) if vue else (d["wd"], d["wf"])
            a, b = max(a, d["wd"]), min(b, d["wf"])
        else:
            a, b = parse_date(s_deb), parse_date(s_fin)
    except (ValueError, StopIteration) as exc:
        return no_update, no_update, no_update, html.Span(str(exc), style={"color": "firebrick"})
    if b < a:
        a, b = b, a
    txt_a, txt_b = f"{a:%Y-%m-%d %H:%M:%S.%f}"[:-3], f"{b:%Y-%m-%d %H:%M:%S.%f}"[:-3]
    return [txt_a, txt_b], txt_a, txt_b, ""


@app.callback(Output("vue", "data"), Input("fig-b", "relayoutData"), Input("cle", "data"),
              prevent_initial_call=True)
def suivre_vue(relayout, cle):
    """Mémorise la plage de temps affichée (zoom / pan), quel que soit le sous-graphe manipulé."""
    if ctx.triggered_id == "cle" or not relayout:
        return None if ctx.triggered_id == "cle" else no_update
    for k, v in relayout.items():
        if re.fullmatch(r"xaxis\d*\.autorange", k) and v:
            return None
        if re.fullmatch(r"xaxis\d*\.range\[0\]", k):
            return [str(v), str(relayout[k.replace("[0]", "[1]")])]
        if re.fullmatch(r"xaxis\d*\.range", k):
            return [str(v[0]), str(v[1])]
    return no_update


@app.callback(Output("fig-psd-sel", "figure"), Output("fig-b", "figure", allow_duplicate=True),
              Input("selection", "data"), State("cle", "data"), prevent_initial_call=True)
def psd_selection(sel, cle):
    if not cle or cle not in MEMOIRE:
        return no_update, no_update
    d = MEMOIRE[cle]
    patch = Patch()
    if not sel:
        patch["layout"]["shapes"] = formes_b(d)
        return fig_vide("Aucune zone sélectionnée", 480), patch
    a, b = parse_date(sel[0]), parse_date(sel[1])
    m = (d["t"] >= np.datetime64(a)) & (d["t"] <= np.datetime64(b))
    duree = (b - a).total_seconds()
    titre = f"PSD sélection : {a:%Y-%m-%d %H:%M:%S} → {b:%H:%M:%S} UTC ({duree:.1f} s, {int(m.sum())} points)"
    fig, _ = psd_ou_message(d["t"][m], d["B"][m], titre, horizontal=True, height=480)
    patch["layout"]["shapes"] = formes_b(d, (a, b))
    return fig, patch


@app.callback(Output("fig-orbite", "figure"), Input("cle", "data"), Input("selection", "data"),
              prevent_initial_call=True)
def orbite(cle, sel):
    if not cle or cle not in MEMOIRE:
        return no_update
    s = (parse_date(sel[0]), parse_date(sel[1])) if sel else None
    return fig_orbite(MEMOIRE[cle], s)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Webapp d'exploration des données MESSENGER")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    print(f"Ouvrir http://127.0.0.1:{args.port}")
    app.run(debug=args.debug, port=args.port)