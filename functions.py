import pandas as pd
import numpy as np
from IPython.display import Markdown as md
from tabulate import tabulate
import re
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


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
        # print("Spectral Type:", spectral_type)
        # print("Luminosity Class:", luminosity_class if luminosity_class else "N/A")

    elif match and 'Ia' in input_str:
        # Extract the spectral type, number, and luminosity class from the matched groups
        spectral_type = f'{match.group(1)}{match.group(2)}'
        luminosity_class = 'Ia'
        # Print the results
        # print("Spectral Type:", spectral_type)
        # print("Luminosity Class:", luminosity_class if luminosity_class else "N/A")

    elif match:
        # Extract the spectral type, number, and luminosity class from the matched groups
        spectral_type = f'{match.group(1)}{match.group(2)}'
        luminosity_class = match.group(3)
        # Print the results
        # print("Spectral Type:", spectral_type)
        # print("Luminosity Class:", luminosity_class if luminosity_class else "N/A")

    else:
        # print("Invalid spectral type format.")
        spectral_type = None
        luminosity_class = None

    return spectral_type, luminosity_class


def extract_number_from_spectral_type(spectral_type):
    pattern = re.compile(r'\d+(\.\d+)?')
    match = pattern.search(spectral_type)
    
    if match and match.group() != '':
        extracted_number = float(match.group())
        
        # Check if the spectral type starts with 'B' and add 10 to the extracted number
        if spectral_type.startswith('B'):
            extracted_number += 10.0
            
        return extracted_number
    else:
        return None


def interpolate_value(spectral_type_values: list, spectral_type_numbers: list, target_number: float):
    # Ensure the input lists are of the same length
    if len(spectral_type_values) != len(spectral_type_numbers):
        raise ValueError("Input lists must have the same length")

    # Create an interpolation function
    interpolate_func = interp1d(spectral_type_numbers, spectral_type_values, kind='linear', fill_value='extrapolate')

    # Use the interpolation function to get the value at the target number
    interpolated_value = interpolate_func(target_number)

    return interpolated_value


def interpolate(df2: pd.DataFrame, spectral_type: str, quantity: str, plot: bool = False):
    """
    
    """
    # Make sure we don't change the input dataframe
    df = df2.copy()

    # Get the short spectral type and luminosity class
    spectral_type_short, luminosity_class = decompose_spectral_type(spectral_type)

    # Make a new column with luminosity class
    df['luminosity_class'] = df['ST'].apply(decompose_spectral_type).apply(lambda x: x[1])

    # If luminosity class equals Ia or Ia+, then set it to I
    if luminosity_class == 'Ia' or 'Ia+':
        luminosity_class = 'I'

    # Filter dataframe for luminosity class
    df = df[df['ST'].str.contains('O')].loc[df['luminosity_class'] == luminosity_class].reset_index(drop=True)

    # Data for the given quantity
    spectral_type_numbers = df['ST'].apply(extract_number_from_spectral_type).tolist()
    quantity_values = df[quantity].tolist()

    # Interpolate
    target_number = extract_number_from_spectral_type(spectral_type_short)
    interpolated_value = interpolate_value(spectral_type_values=quantity_values, spectral_type_numbers=spectral_type_numbers, target_number=target_number)

    if plot:
        plt.plot(spectral_type_numbers, quantity_values, color='blue')
        plt.scatter([target_number], [interpolated_value], color='orange')
        plt.ylabel(quantity)
        plt.xlabel('Spectral type')
        plt.grid(True)
        plt.show()

    return interpolated_value