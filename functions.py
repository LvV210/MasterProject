import pandas as pd
import numpy as np
from IPython.display import Markdown as md
from tabulate import tabulate
import re
import math
from astropy.constants import R_sun, L_sun, sigma_sb
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import tarfile
import os
import sympy as sp


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
    if luminosity_class == 'Ia' or 'Ia+' or 'Ib' or 'Ib+':
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


def evolutionary_track(Z: float, Y: float, M: str, plot_all: bool = False, plot_single: bool = False):
    """
    
    """
    
    # Folder path to the data extracted from .gz.tar file
    folder_path = f'evolutionary_tracks/extract/Z{Z}Y{Y}/'
    # Specify the path to your tar.gz file
    file_path = f'evolutionary_tracks/Z{Z}Y{Y}.tar.gz'

    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        print(f"Folder |{folder_path}| already exists.")
    else:
        print(f"Extracting files from {folder_path}...")
        # Open the tar.gz file for reading
        with tarfile.open(file_path, 'r:gz') as tar:
            # Extract all contents to a specific directory (optional)
            tar.extractall(path='evolutionary_tracks/extract')

            # List the contents of the tar.gz file
            file_names = tar.getnames()
            print(f"LOADED contents of {file_path}.")


    # List all files in the folder
    file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and 'ADD' not in f and '.HB' not in f]

    # Plot evolutionary track of all stars
    if plot_all:
        plt.figure()
        for file_name in file_names:
            df_evolutionary_track = pd.read_csv(f'evolutionary_tracks/extract/Z{Z}Y{Y}/{file_name}', delim_whitespace=True)

            logL = df_evolutionary_track["LOG_L"].tolist()
            logT = df_evolutionary_track["LOG_TE"].tolist()

            logL = [float(x) for x in logL]
            logT = [float(x) for x in logT]

            plt.plot(logT, logL)

        plt.grid(True)
        plt.title(f"All evolutionary tracks for Z={Z}, Y={Y}, M={float(M)}"+ r"$M_{\odot}$")
        plt.gca().invert_xaxis()
        plt.show()

    # Evolutionary track of a single star of mass M
    df_evolutionary_track = pd.read_csv(f'evolutionary_tracks/extract/Z{Z}Y{Y}/Z{Z}Y{Y}OUTA1.74_F7_M{M}.DAT', delim_whitespace=True)
    if plot_single:

        logL = df_evolutionary_track["LOG_L"].tolist()
        logT = df_evolutionary_track["LOG_TE"].tolist()

        logL = [float(x) for x in logL]
        logT = [float(x) for x in logT]

        plt.figure()
        plt.title(f"Evolutionary track for Z={Z}, Y={Y}, M={float(M)}"+ r"$M_{\odot}$")
        plt.xlabel(r"$log(T_{eff})$")
        plt.ylabel(r"$log(\frac{L}{L_{\odot}})$")
        plt.plot(logT, logL)
        plt.grid(True)
        plt.gca().invert_xaxis()
        plt.show()
    
    return df_evolutionary_track


def end_main_sequence(lst: list):
    """
    Finds where the first 0 value is in a list
    """
    for i, value in enumerate(lst):
        if value == 0:
            return i
    return None


def find_ZAMS_index(H: list):
    """
    Find the index where the Zero Age Main-Sequence is in the evolutionary track.
    The ZAMS is where the hydrogen fraction in the core starts to decrease i.e. hydrogen burning has started.


    Args:
        H (list): Hydrogen fraction as function of time

    Returns:
        [int]: Index of start ZAMS
    """
    H_start = H[0]
    for i in range(1, len(H)):
        if H[i] < (H_start - 0.01):
            return i - 1
    return -1  # If the list is entirely non-decreasing


def ZAMS(Z: float, Y: float):
    """
    Find the ZAMS, based on the start of hydrogen burning in the core for the different stellar masses

    Args:
        Z (float): Metallicity
        Y (float): Helium fraction

    Returns:
        L_ZAMS (list): List of log(L) values for the ZAMS
        T_ZAMS (list): List of log(T) values for the ZAMS
    """
    # List to save ZAMS
    L_ZAMS = []
    T_ZAMS = []

    # List for all masses to calc ZAMS for
    M_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 24, 28, 30, 35, 40, 50, 55, 60, 65, 70, 80, 100, 120, 150]
    M_values = ['{:07.3f}'.format(m) for m in M_values]

    for M in M_values:
        df = evolutionary_track(Z=Z, Y=Y, M=M)

        ZAMS_index = find_ZAMS_index(df['H_CEN'].tolist())
        logL = df["LOG_L"].tolist()[ZAMS_index]
        logT = df["LOG_TE"].tolist()[ZAMS_index]

        L_ZAMS.append(logL)
        T_ZAMS.append(logT)

    return L_ZAMS, T_ZAMS


def time_intervals(start_point, N_iterations, N_intervals):
    # Define the range and number of intervals
    # Calculate the step size
    step_size = (N_iterations - start_point) / (N_intervals - 1)
    # Generate the list of equally spaced numbers
    equally_spaced_numbers = [round((start_point + i * step_size - 1), 0) for i in range(N_intervals)]
    return equally_spaced_numbers


"""
ERROR ANALYSIS
"""
def extinction_and_error(Rh_, SigmaRh_, JHobs_, SigmaJHobs_, JH0_, SigmaJH0_):
    # Define the symbols
    Rh, SigmaRh, JHobs, SigmaJHobs, JH0, SigmaJH0 = sp.symbols('Rh SigmaRv JHobs SigmaJHobs JH0 SigmaJH0')

    # Define the function
    Ah = Rh * (JHobs - JH0)

    # Calculate the partial derivatives
    partial_derivative_Rh = sp.diff(Ah, Rh)
    partial_derivative_JHobs = sp.diff(Ah, JHobs)
    partial_derivative_JH0 = sp.diff(Ah, JH0)

    # Calculate the error expression
    error_Ah = sp.sqrt(
        (partial_derivative_Rh * SigmaRh)**2 +
        (partial_derivative_JHobs * SigmaJHobs)**2 +
        (partial_derivative_JH0 * SigmaJH0)**2)
    
    # Substitute values
    values = {
        Rh: Rh_,
        SigmaRh: SigmaRh_,
        JHobs: JHobs_,
        SigmaJHobs: SigmaJHobs_,
        JH0: JH0_,
        SigmaJH0: SigmaJH0_
    }

    Ah_value = Ah.subs(values).evalf()
    Ah_error = error_Ah.subs(values).evalf()

    return Ah_value, Ah_error


def luminosity_error_function():
    # Define the symbols
    BCh, mh, d, Ah, SigmaBCh, Sigmad, Sigmamh, SigmaAh = sp.symbols(
        'BCh mh d Ah SigmaBCh Sigmad Sigmamh SigmaAh'
    )

    # Define the function L
    L = 10**(-0.4*(-4.74 + BCh + mh - Ah - (5*sp.log(d))/sp.log(10) + 5))

    # Calculate the partial derivatives
    partial_derivative_BCh = sp.diff(L, BCh)
    partial_derivative_mh = sp.diff(L, mh)
    partial_derivative_d = sp.diff(L, d)
    partial_derivative_Ah = sp.diff(L, Ah)

    # Calculate the error expression
    error_L = sp.sqrt(
        (partial_derivative_BCh * SigmaBCh)**2 +
        (partial_derivative_mh * Sigmamh)**2 +
        (partial_derivative_d * Sigmad)**2 +
        (partial_derivative_Ah* SigmaAh)**2)

    return error_L

def luminosity_error(BCh_: float, SigmaBCh_: float, mh_: float, Sigmamh_: float, 
                     d_: float, Sigmad_: float, Ah_: float, SigmaAh_: float):

    # Calculate error function
    error_L = luminosity_error_function()

    # Define the symbols
    BCh, mh, d, Ah, SigmaBCh, Sigmad, Sigmamh, SigmaAh = sp.symbols(
        'BCh mh d Ah SigmaBCh Sigmad Sigmamh SigmaAh'
    )

    # Define the function L
    L = 10**(-0.4*(-4.74 + BCh + mh - Ah - (5*sp.log(d))/sp.log(10) + 5))

    # Substitute values
    values = {
        BCh: BCh_,
        mh: mh_,
        d: d_,
        Ah: Ah_,
        SigmaBCh: SigmaBCh_,
        Sigmad: Sigmad_,
        Sigmamh: Sigmamh_,
        SigmaAh: SigmaAh_,
    }

    # Calculate the error expression with values substituted
    error_L_with_values = error_L.subs(values)
    # Calculate the numerical value
    numerical_value_error_L = error_L_with_values.evalf()

    return numerical_value_error_L

def expected_radius_error(L_, SigmaL_, Teff_, SigmaTeff_):
    # Define the symbols
    L, SigmaL, Teff, SigmaTeff = sp.symbols('L SigmaL Teff SigmaTeff')

    # Define the function
    R = ((L_sun.value / R_sun.value**2) * (L / (4 * np.pi * sigma_sb.value * Teff**4)))**(1/2)

    # Calculate the partial derivatives
    partial_derivative_L = sp.diff(R, L)
    partial_derivative_Teff = sp.diff(R, Teff)

    # Calculate the error expression
    error_R = sp.sqrt(
        (partial_derivative_L * SigmaL)**2 +
        (partial_derivative_Teff * SigmaTeff)**2)
    
    # Substitute values
    values = {
        L: L_,
        SigmaL: SigmaL_,
        Teff: Teff_,
        SigmaTeff: SigmaTeff_
    }

    R_value = R.subs(values).evalf()
    R_error = error_R.subs(values).evalf()

    return R_value, R_error


def Teff_error(ST):
    # Decompose spectral type
    spectral_type, luminosity_class = decompose_spectral_type(ST)
    # If luminosity class equals Ia or Ia+, then set it to I
    if luminosity_class == 'Ia' or 'Ia+' or 'Ib' or 'Ib+':
        luminosity_class = 'I'
    # Return appropriate error
    if luminosity_class == 'I':
        return 1446
    elif luminosity_class == 'III':
        return 529
    elif luminosity_class == 'V':
        return 1021