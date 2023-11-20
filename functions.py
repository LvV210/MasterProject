import pandas as pd
import numpy as np
from IPython.display import Markdown as md
from tabulate import tabulate
import re


def scientific_notation(df: pd.DataFrame):

    df = df.applymap(lambda x: "{:.2E}".format(x) if isinstance(x, (int, float, np.floating)) else x)

    return df


def display_df(df2: pd.DataFrame, scientific_notation: bool = True, round: bool = False):
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
                "distance": r"$d[pc]$",
                "distanceBJ": r"$d_{BJ}[pc]$",
                "luminosity": r"$L_{ob}[L_{\odot}]$",
                "(B-V)obs": r"$(B-V)_{obs}$",
                "(B-V)0": r"$(B-V)_{0}$",
                "BC": r"$BC_{v}$",
                "mv": r"$m_{v}$",
                "Teff": r"$T_{eff}[K]$",
                "Mv": r"$M_{v}$",
                "log(g_spec)": r"$log(g_{spec})$",
                "log(L/Lsun)": r"$log(\frac{L}{L_{\odot}}$",
                "M_spec": r"$M_{spec}$"}
    
    # Rename columns
    df = df.rename(columns=columns)

    # Scientific notation
    if scientific_notation:
        df = df.applymap(lambda x: "*" + "%.2E" % x if isinstance(x, (int, float, np.floating)) else x)

    # Round
    if round:
        df = df.applymap(lambda x: round(x, 1) if isinstance(x, (int, float, np.floating)) else x)

    # To markdown
    markdown_text = df.to_markdown(index=False, tablefmt="pipe")

    # Remove '*' from text
    markdown_text = markdown_text.replace('*', '')

    return md(markdown_text)


def decompose_spectral_type(input_str: str):
    """
    Need:
        import re
    Example usage:
        spectral_type_input = "O6Ia+"
        decompose_spectral_type(spectral_type_input)

    """
    # Define a regular expression pattern to match the spectral type
    # The pattern consists of a letter (A to Z), optionally followed by a number and/or a luminosity class
    pattern = re.compile(r'([A-Z]+)(\d*\.\d+|\d+)?([IV]+[+]?)?')

    # Use the pattern to search for matches in the input string
    match = pattern.search(input_str)

    if match and 'Ia+' in input_str:
        # Extract the spectral type, number, and luminosity class from the matched groups
        spectral_type = f'{match.group(1)}{match.group(2)}'
        luminosity_class = 'Ia+'
        # Print the results
        print("Spectral Type:", spectral_type)
        print("Luminosity Class:", luminosity_class if luminosity_class else "N/A")

    elif match and 'Ia' in input_str:
        # Extract the spectral type, number, and luminosity class from the matched groups
        spectral_type = f'{match.group(1)}{match.group(2)}'
        luminosity_class = 'Ia'
        # Print the results
        print("Spectral Type:", spectral_type)
        print("Luminosity Class:", luminosity_class if luminosity_class else "N/A")

    elif match:
        # Extract the spectral type, number, and luminosity class from the matched groups
        spectral_type = f'{match.group(1)}{match.group(2)}'
        luminosity_class = match.group(3)
        # Print the results
        print("Spectral Type:", spectral_type)
        print("Luminosity Class:", luminosity_class if luminosity_class else "N/A")

    else:
        print("Invalid spectral type format.")
        spectral_type = None
        luminosity_class = None

    return spectral_type, luminosity_class