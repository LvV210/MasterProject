import pandas as pd
from IPython.display import Markdown as md

def scientific_notation(df):
    df2 = df.copy()
    # Get the number of rows and columns in the DataFrame
    num_rows, num_columns = df2.shape

    # Double for loop to iterate through all entries
    for i in range(num_rows):
        for j in range(num_columns):
            # Check if var1 is a number
            if isinstance(df2.loc[i,j], (int, float, complex)):
                df2.loc[i,j] = "{:.2E}".format(df2.loc[i,j])

    return df2

def display_df(df: pd.DataFrame):
    """
    Displays a given dataframe in a nice way with better collumn names and units.
    """
    columns = {"R_true": (r"$R_{true}[R_{\odot}]$"),
                "R_expected": r"$R_{expected}[R_{\odot}]$",
                "R_true/R_expected": r"$\frac{R_{true}}{R_{expected}}$",
                "L_true": r"$L_{true}[L_{\odot}]$",
                "L_expected": r"$L_{expected}[L_{\odot}]$",
                "L_true/L_expected": r"$\frac{L_{true}}{L_{expected}$}",
                "ST": "Spec. Type",
                "ST_short": "Spec. Type (short)",
                "period": r"$P[units]$",
                "spinperiod": r"$P_{spin}[units]$",
                "eclipseduration": r"eclipseduration[units]$",
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

    return md(df.rename(columns=columns).to_markdown(index=False))
