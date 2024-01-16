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
import subprocess
from astropy.io import fits



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


"""
Extract numbers and errors from ##(##) notation
"""
def count_decimal_places(number):
    # Convert the number to a string
    num_str = str(number)
    
    # Check if the string contains a decimal point
    if '.' in num_str:
        # Find the index of the decimal point
        decimal_index = num_str.index('.')
        
        # Count the number of characters after the decimal point
        decimal_places = len(num_str) - decimal_index - 1
        
        return decimal_places
    else:
        # If there is no decimal point, return 0
        return 0



def parse_decimal_number_string(s):
    match = re.match(r'(\d+\.\d+)\((\d+)\)', s)
    if match:
        value = float(match.group(1))
        decimal_places_value = count_decimal_places(value)
        decimal_places_err = count_decimal_places(float('0.' + match.group(2)))
        err = float('0.' + (decimal_places_value - (decimal_places_err - 1) - 1) * '0' + match.group(2))
        return value, err
    else:
        raise ValueError("Invalid number string format")



def extract_numbers_error_notation(string):
    # Define a regular expression pattern to match the number and the number in brackets
    pattern = re.compile(r'(\d+\.\d+|\d+)(\(\d+\))?')

    # Try to match the pattern in the given string
    match = pattern.match(string)

    if match:
        # Extract the main number and the number in brackets
        main_number_str = match.group(1)
        main_number = float(main_number_str) if '.' in main_number_str else int(main_number_str)
        
        number_in_brackets = int(match.group(2)[1:-1]) if match.group(2) else None

        return str(main_number), str(number_in_brackets)
    else:
        # If no match is found, return None for both values
        return None, None



def extract_number_and_error(string):

    value, err = extract_numbers_error_notation(string)

    # For integers
    if float(value) == int(float(value)):
        return value, err

    # For decimal numbers
    elif float(value) != int(float(value)):
        value, err = parse_decimal_number_string(string)
        return value, err



"""
Spectral Type
"""

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
    folder_path = f'../evolutionary_tracks/extract/Z{Z}Y{Y}/'
    # Specify the path to your tar.gz file
    file_path = f'../evolutionary_tracks/Z{Z}Y{Y}.tar.gz'

    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        print(f"Folder |{folder_path}| already exists.")
    else:
        print(f"Extracting files from {folder_path}...")
        # Open the tar.gz file for reading
        with tarfile.open(file_path, 'r:gz') as tar:
            # Extract all contents to a specific directory (optional)
            tar.extractall(path='../evolutionary_tracks/extract')

            # List the contents of the tar.gz file
            file_names = tar.getnames()
            print(f"LOADED contents of {file_path}.")


    # List all files in the folder
    file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and 'ADD' not in f and '.HB' not in f]

    # Plot evolutionary track of all stars
    if plot_all:
        plt.figure()
        for file_name in file_names:
            df_evolutionary_track = pd.read_csv(f'../evolutionary_tracks/extract/Z{Z}Y{Y}/{file_name}', delim_whitespace=True)

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
    df_evolutionary_track = pd.read_csv(f'../evolutionary_tracks/extract/Z{Z}Y{Y}/Z{Z}Y{Y}OUTA1.74_F7_M{M}.DAT', delim_whitespace=True)
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


def luminosity_error_asymmetric(BCh_: float, SigmaBCh_: float, mh_: float, Sigmamh_: float, 
                     d_: float, Sigmad_plus: float, Sigmad_minus: float, Ah_: float, SigmaAh_: float):

    # Define the symbols
    BCh, mh, d, Ah, SigmaBCh, Sigmadplus, Sigmadminus, Sigmamh, SigmaAh = sp.symbols(
        'BCh mh d Ah SigmaBCh Sigmadplus Sigmadminus Sigmamh SigmaAh'
    )

    # Define the function L
    L = 10**(-0.4*(-4.74 + BCh + mh - Ah - (5*sp.log(d))/sp.log(10) + 5))

    # Calculate the partial derivatives
    partial_derivative_BCh = sp.diff(L, BCh)
    partial_derivative_mh = sp.diff(L, mh)
    partial_derivative_d = sp.diff(L, d)
    partial_derivative_Ah = sp.diff(L, Ah)

    # Calculate the error expression
    error_L_plus = sp.sqrt(
        (partial_derivative_BCh * SigmaBCh)**2 +
        (partial_derivative_mh * Sigmamh)**2 +
        (partial_derivative_d * Sigmadplus)**2 +
        (partial_derivative_Ah * SigmaAh)**2)

    # Calculate the error expression
    error_L_minus = sp.sqrt(
        (partial_derivative_BCh * SigmaBCh)**2 +
        (partial_derivative_mh * Sigmamh)**2 +
        (partial_derivative_d * Sigmadminus)**2 +
        (partial_derivative_Ah * SigmaAh)**2)

    # Substitute values
    values = {
        BCh: BCh_,
        mh: mh_,
        d: d_,
        Ah: Ah_,
        SigmaBCh: SigmaBCh_,
        Sigmadplus: Sigmad_plus,
        Sigmadminus: Sigmad_minus,
        Sigmamh: Sigmamh_,
        SigmaAh: SigmaAh_,
    }

    # Calculate the error expression with values substituted
    error_L_with_values_plus = error_L_plus.subs(values)
    error_L_with_values_minus = error_L_minus.subs(values)
    # Calculate the numerical value
    numerical_value_error_L_plus = error_L_with_values_plus.evalf()
    numerical_value_error_L_minus = error_L_with_values_minus.evalf()

    return numerical_value_error_L_minus, numerical_value_error_L_plus



def expected_radius_error_asymmetric(L_, SigmaL_plus, SigmaL_minus, Teff_, SigmaTeff_):
    # Define the symbols
    L, SigmaLplus, SigmaLminus, Teff, SigmaTeff = sp.symbols('L SigmaLplus SigmaLminus Teff SigmaTeff')

    # Define the function
    R = ((L_sun.value / R_sun.value**2) * (L / (4 * np.pi * sigma_sb.value * Teff**4)))**(1/2)

    # Calculate the partial derivatives
    partial_derivative_L = sp.diff(R, L)
    partial_derivative_Teff = sp.diff(R, Teff)

    # Calculate the error expression
    error_R_plus = sp.sqrt(
        (partial_derivative_L * SigmaLplus)**2 +
        (partial_derivative_Teff * SigmaTeff)**2)

    error_R_minus = sp.sqrt(
        (partial_derivative_L * SigmaLminus)**2 +
        (partial_derivative_Teff * SigmaTeff)**2)
    
    # Substitute values
    values = {
        L: L_,
        SigmaLplus: SigmaL_plus,
        SigmaLminus: SigmaL_minus,
        Teff: Teff_,
        SigmaTeff: SigmaTeff_
    }

    R_value = R.subs(values).evalf()
    R_error_plus = error_R_plus.subs(values).evalf()
    R_error_minus = error_R_minus.subs(values).evalf()

    return R_value, R_error_plus, R_error_minus

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
    






"""
UVES spectra analysis
"""
def extract_spectrum_within_range(wavelengths, flux, start_wavelength, end_wavelength):
    """
    Extract wavelength and flux values within a given range.

    Parameters:
    - wavelengths: List or array of wavelength values.
    - flux: List or array of flux values.
    - start_wavelength: Lower bound of the wavelength range.
    - end_wavelength: Upper bound of the wavelength range.

    Returns:
    - extracted_wavelengths: Wavelength values within the specified range.
    - extracted_flux: Flux values corresponding to the selected wavelengths.
    """

    # Find indices corresponding to the specified wavelength range
    indices = np.where((wavelengths >= start_wavelength) & (wavelengths <= end_wavelength))[0]

    # Extract wavelength and flux values within the range
    extracted_wavelengths = wavelengths[indices]
    extracted_flux = flux[indices]

    return extracted_wavelengths, extracted_flux



def divide_colormap(num_parts, colormap_name='rainbow'):
    cmap = plt.get_cmap(colormap_name)
    num_colors = cmap.N
    colors = [cmap(i / num_colors) for i in range(0, num_colors, num_colors // num_parts)]
    return colors



def import_spectra(object: str):

    # Path to root folder
    folder_path = f"../Spectra/{object}/"

    # Get a list of all files in the folder
    all_files = os.listdir(folder_path)

    # Filter files that start with "ADP"
    adp_files = [file for file in all_files if file.startswith("ADP")]

    # Gather the spectra
    spectra = []
    for file in adp_files:
        data = fits.getdata(folder_path + file)
        wavelength = data['WAVE'][0]
        flux = data['FLUX'][0]
        spectra.append((wavelength, flux))

    return spectra



def plot_spectrum(object: str):

    # Import spectra
    spectra = import_spectra(object)

    # Spectral lines and wavelengths in angstrom
    spectral_lines = [
        (4026, "HeI+II 4026"),
        (4200, "HeII 4200"),
        (4686, "HeII 4686"),
        (4634, "NIII 4634-40-42 (emission)"),
        (4144, "HeI 4144"),
        (4388, "HeI 4388"),
        (4542, "HeII 4542"),
        (4552, "SiII 4552"),
        (4686, "HeII 4686"),
        (4861, "Hb 4861"),
        (5016, "HeI 5016"),
        (5876, "HeI 5876"),
        (5890, "NaI 5890"),
        (5896, "NaI 5896"),
        (6527, "HeII 6527"),
        (6563, "Ha 6563")
    ]

    spectral_lines = [
        (4026, "He I + II 4026"),
        (4200, "He II 4200"),
        (4634, "N III 4634-40-42 (emission)"),
        (4686, "He II 4686"),
        (4144, "He I 4144"),
        (4388, "He I 4388"),
        (4541, "He II 4541"),
        (4552, "Si II 4552"),
        (4686, "He II 4686"),
        (4861, "Hb 4861"),
        (5016, "He I 5016"),
        (5876, "He I 5876"),
        (5890, "Na I 5890"),
        (5896, "Na I 5896"),
        (6527, "He II 6527"),
        (6563, "Ha 6563"),
        (4471, "He I 4471"),
        (4058, "N IV 4058"),
        (4116, "Si IV 4116"),
        (4097, "N III 4097"),
        (4504, "Si IV 4686-4504"),
        (4713, "He I 4713"),
        (4187, "C III 4187"),
        (4121, "He I 4121"),
        (3995, "N II 3995"),
        (4350, "O II 4350"),
        (4128, "Si I 4128-30"),
        (4481, "Mg II 4481"),
        (4233, "Fe II 4233"),
    ]

    colors = divide_colormap(len(spectral_lines))

    # Plot the spectrum
    plt.figure(figsize=(10, 5))
    for spectrum in spectra:
        plt.plot(spectrum[0], spectrum[1] / max(spectrum[1]))

    # Plot wavelengths of spectral lines
    for wavelength, label in spectral_lines:
        plt.vlines(wavelength, -1000, 1000, label=label, color=colors[spectral_lines.index((wavelength, label))])

    plt.xlabel('Wavelength (Angstroms)')
    plt.ylabel('Flux')
    plt.title(f'Spectrum {object}')
    plt.ylim(0, 1.2)
    plt.legend()
    plt.show()

    return