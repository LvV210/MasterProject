import pandas as pd
import numpy as np
from IPython.display import Markdown as md
from tabulate import tabulate


def scientific_notation(df: pd.DataFrame):

    df = df.applymap(lambda x: "{:.2E}".format(x) if isinstance(x, (int, float, np.floating)) else x)

    return df


def display_df(df2: pd.DataFrame):
    """
    Displays a given dataframe in a nice way with better collumn names and units.
    """
    df = df2.copy()
    columns = {"R_true": (r"$R_{true}[R_{\odot}]$"),
                "R_expected": r"$R_{expected}[R_{\odot}]$",
                "R_true/R_expected": r"$\frac{R_{true}}{R_{expected}}$",
                "L_true": r"$L_{true}[L_{\odot}]$",
                "L_expected": r"$L_{expected}[L_{\odot}]$",
                "L_true/L_expected": r"$\frac{L_{true}}{L_{expected}}$",
                "ST": "Spec. Type",
                "ST_short": "Spec. Type (short)",
                "period": r"$P[days]$",
                "spinperiod": r"$P_{spin}[s]$",
                "eclipseduration": r"eclipseduration[units]",
                "RV": r"$v_{rad}[km s^{-1}]$",
                "Mob": r"$M_{ob}[M_{\odot}]$",
                "Mx": r"$M_{x} [M_{\odot}]$",
                "Rob": r"$R_{ob}[R_{\odot}]$",
                "Rx": r"$R_{x}[R_{\odot}]$",
                "parallax": r"$p'' ['']$",
                "errparallax": r"$\delta p''['']$",
                "distance": r"$d[kpc]$",
                "distanceBJ": r"$d_{BJ}[kpc]$",
                "luminosity": r"$L_{ob}[L_{\odot}]$",
                "(B-V)obs": r"$(B-V)_{obs}$",
                "(B-V)0": r"$(B-V)_{0}$",
                "BC": r"$BC_{v}$",
                "mv": r"$m_{v}$",
                "Teff": r"$T_{eff}[K]$"}
    
    # Rename columns
    df = df.rename(columns=columns)

    # Scientific notation
    df = df.applymap(lambda x: "*" + "%.2E" % x if isinstance(x, (int, float, np.floating)) else x)

    # To markdown
    markdown_text = df.to_markdown(index=False, tablefmt="pipe")

    # Remove '*' from text
    markdown_text = markdown_text.replace('*', '')

    return md(markdown_text)
