"""
Point de départ speasy + AMDA pour les données MAG de MESSENGER.

Prérequis : Python >= 3.10
    python -m pip install speasy scipy matplotlib

Usage :
    python messenger_amda_speasy.py
        -> liste les paramètres AMDA dont l'id ou le nom contient "mes"
    python messenger_amda_speasy.py <id_param> <debut> <fin> [<debut_mva> <fin_mva>]
        -> charge B, calcule |B|, fait une MVA (repère LMN), filtre et trace
    ex : python messenger_amda_speasy.py <id_B_MSO> 2008-01-14T18:00 2008-01-14T20:00

Les temps sont en UTC, l'intervalle est semi-ouvert [debut, fin).
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

import speasy as spz
from speasy.signal.filtering import sosfiltfilt
from speasy.signal.resampling import resample


# --- 1. Trouver les identifiants de produits -------------------------------
def chercher_parametres(mot_cle, provider="amda"):
    """Renvoie {id: nom} des paramètres dont l'id ou le nom contient mot_cle."""
    inv = getattr(spz.inventories.flat_inventories, provider).parameters
    mot = mot_cle.lower()
    return {
        pid: idx.spz_name()
        for pid, idx in inv.items()
        if mot in pid.lower() or mot in idx.spz_name().lower()
    }


# --- 2. Charger et nettoyer ------------------------------------------------
def charger(param_id, debut, fin):
    """Télécharge un paramètre AMDA (mis en cache localement ensuite)."""
    var = spz.get_data(f"amda/{param_id}", debut, fin)
    if var is None:
        raise RuntimeError(f"Aucune donnée pour {param_id} entre {debut} et {fin}")
    # Les valeurs manquantes sont codées par une sentinelle (FILLVAL) -> NaN
    return var.replace_fillval_by_nan()


def norme(var):
    """|B| (reste NaN là où une composante manque)."""
    return np.sqrt(np.sum(var.values[:, :3] ** 2, axis=1))


# --- 3. Analyse en variance minimale (repère LMN) --------------------------
def mva(var):
    """
    Analyse en variance minimale sur les 3 composantes de B.
    Renvoie (R, valeurs_propres) où les lignes de R sont L, M, N
    (variance max, intermédiaire, min). B_lmn = B_xyz @ R.T
    Un rapport lambda_M / lambda_N > ~5-10 indique une normale bien définie.
    """
    b = var.values[:, :3]
    b = b[np.all(np.isfinite(b), axis=1)]
    valp, vecp = np.linalg.eigh(np.cov(b, rowvar=False))  # ordre croissant
    R = vecp[:, ::-1].T                                   # L, M, N
    return R, valp[::-1]


# --- 4. Filtrage passe-bande ----------------------------------------------
def passe_bande(var, periode_min_s, periode_max_s, dt_s=1.0, ordre=4):
    """
    Ré-échantillonne à pas constant (dt_s) puis filtre (aller-retour, sans
    déphasage). Les lignes NaN sont retirées avant (un seul NaN contaminerait
    tout le signal filtré), donc le ré-échantillonnage interpole à travers les
    trous : à garder en tête si l'intervalle contient de longues lacunes.
    """
    reg = resample(var.sanitized(), np.timedelta64(int(dt_s * 1e3), "ms"))
    sos = signal.butter(ordre, [1 / periode_max_s, 1 / periode_min_s],
                        btype="bandpass", fs=1 / dt_s, output="sos")
    return sosfiltfilt(sos, reg)


# --- 5. Script principal -----------------------------------------------------
def main(argv):
    if len(argv) < 4:
        res = chercher_parametres("mes")
        print(f"{len(res)} paramètres AMDA contenant 'mes' :")
        for pid, nom in sorted(res.items()):
            print(f"  {pid:40s} {nom}")
        print("\nRelance avec : <id_param> <debut> <fin> [<debut_mva> <fin_mva>]")
        return

    param_id, debut, fin = argv[1:4]
    b = charger(param_id, debut, fin)
    print(f"{len(b)} points, colonnes {b.columns}, unité {b.unit}")

    # MVA sur un sous-intervalle (la traversée de magnétopause) si fourni
    sous = b
    if len(argv) >= 6:
        sous = b[np.datetime64(argv[4]):np.datetime64(argv[5])]
    R, lam = mva(sous)
    print("L, M, N (dans le repère des données) :\n", np.round(R, 3))
    print(f"lambda_L : lambda_M : lambda_N = {lam.round(2)}  "
          f"(lambda_M/lambda_N = {lam[1] / lam[2]:.1f})")
    b_lmn = b.values[:, :3] @ R.T

    filtre = passe_bande(b, periode_min_s=5, periode_max_s=60)  # bande à adapter

    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True, layout="constrained")
    for i, c in enumerate(b.columns[:3]):
        ax[0].plot(b.time, b.values[:, i], lw=0.6, label=c)
    ax[0].plot(b.time, norme(b), "k", lw=0.8, label="|B|")
    ax[0].set_ylabel(f"B [{b.unit}]")
    for i, c in enumerate("LMN"):
        ax[1].plot(b.time, b_lmn[:, i], lw=0.6, label=f"B_{c}")
    ax[1].set_ylabel("B LMN")
    for i, c in enumerate(b.columns[:3]):
        ax[2].plot(filtre.time, filtre.values[:, i], lw=0.6, label=c)
    ax[2].set_ylabel("B filtré 5-60 s")
    for a in ax:
        a.legend(loc="upper right", fontsize=8)
        a.grid(alpha=0.3)
    fig.savefig("messenger_b.png", dpi=150)

    b.to_dataframe().to_csv("messenger_b.csv")
    print("Écrit : messenger_b.png, messenger_b.csv")

if __name__ == "__main__":
    main(sys.argv)

