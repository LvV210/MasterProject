import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import json

from typing import Iterable
from scipy.interpolate import CubicSpline
from PyAstronomy import pyasl
from scipy.optimize import curve_fit

from functions import extract_spectrum_within_range


"""
Functions that are used for model fitting
"""

def import_model_callib(file_path: str)->dict:
    """
    Import the model at given file path

    Args:
        file_path (str): Exact path to the file with the model

    Returns:
        dict: Dictionary with WAVELENGTH and FLUX
    """

    # Open the text file for reading
    with open(file_path, 'r') as file:
        # Read all lines from the file
        lines = file.readlines()

    spectrum = {'WAVELENGTH': [], 'FLUX': []}
    # Process the lines as needed
    for line in lines:
        # Perform operations on each line
        # For example, you might split the line into values if it's a space-separated list of numbers
        values = line.split()
        # Process the values accordingly
        spectrum['WAVELENGTH'].append(float(values[0]))
        spectrum['FLUX'].append(10 ** (float(values[1])))

    return spectrum



def import_model(file_path: str)->dict:
    """
    Import the model at given file path

    Args:
        file_path (str): Exact path to the file with the model

    Returns:
        dict: Dictionary with WAVELENGTH and FLUX
    """

    # Open the text file for reading
    with open(file_path, 'r') as file:
        # Read all lines from the file
        lines = file.readlines()

    spectrum = {'WAVELENGTH': [], 'FLUX': []}
    # Process the lines as needed
    for line in lines:
        # Perform operations on each line
        # For example, you might split the line into values if it's a space-separated list of numbers
        values = line.split()
        # Process the values accordingly
        spectrum['WAVELENGTH'].append(float(values[0]))
        spectrum['FLUX'].append((float(values[1])))

    return spectrum



def import_models_callib(galaxy: str)->dict:
    """
    Imports all models for a given galaxy

    Args:
        galaxy (str): Galaxy to import the models for (Milkyway, SMC, LMC)

    Returns:
        models (dict): Dictionary with models. Name: T{Teff}logg{log(g)}
    """

    # Path to the models
    model_path = '/mnt/c/Users/luukv/Documenten/NatuurSterrkenkundeMasterProject/CodeMP/MasterProject/ModelFitting/Models/'

    # Assign folder name corresponding to the galaxy
    if galaxy == 'Milkyway':
        folder_name = model_path + 'gal-ob-i_line_calib_all'
    elif galaxy == 'SMC':
        folder_name = model_path + 'smc-ob-i_line_calib_all'
    elif galaxy == 'LMC':
        folder_name = model_path + 'lmc-ob-i_line_calib_all'

    # Get the list of all filenames in the folder
    file_names = [f for f in os.listdir(folder_name) if os.path.isfile(os.path.join(folder_name, f)) and f != 'modelparameters.txt']

    # Define a regex pattern to match the two numbers for T and log(g) from file name
    pattern = re.compile(r'(\d+)-(\d+)')

    # Import all spectra and add them to a dictionary
    models = {}
    for file in file_names:
        # Extract temperature and log(g) from file name
        T, log_g = map(int, pattern.findall(file)[0])
        # Get spectrum
        spectrum = import_model(folder_name + '/' + file)
        # Save spectrum
        model = {}
        model['Name'] = f'T{T * 1000}logg{log_g / 10}'
        model['WAVELENGTH'] = spectrum['WAVELENGTH']
        model['FLUX'] = spectrum['FLUX']
        model['Teff'] = T * 1000
        model['log(g)'] = log_g / 10

        models[f'T{T * 1000}logg{log_g / 10}'] = model

    return models



def import_models(galaxy: str)->dict:
    """
    Imports all models for a given galaxy

    Args:
        galaxy (str): Galaxy to import the models for (Milkyway, SMC, LMC)

    Returns:
        models (dict): Dictionary with models. Name: T{Teff}logg{log(g)}
    """

    # Path to the models
    model_path = '/mnt/c/Users/luukv/Documenten/NatuurSterrkenkundeMasterProject/CodeMP/MasterProject/ModelFitting/Models/'

    # Assign folder name corresponding to the galaxy
    if galaxy == 'Milkyway':
        folder_name = model_path + 'gal-ob-i_line_all'
    elif galaxy == 'SMC':
        folder_name = model_path + 'smc-ob-i_line_all'
    elif galaxy == 'LMC':
        folder_name = model_path + 'lmc-ob-i_line_all'

    # Get the list of all filenames in the folder
    file_names = [f for f in os.listdir(folder_name) if os.path.isfile(os.path.join(folder_name, f)) and f != 'modelparameters.txt']

    # Define a regex pattern to match the two numbers for T and log(g) from file name
    pattern = re.compile(r'(\d+)-(\d+)')

    # Import all spectra and add them to a dictionary
    models = {}
    for file in file_names:
        # Extract temperature and log(g) from file name
        T, log_g = map(int, pattern.findall(file)[0])
        # Get spectrum
        spectrum = import_model(folder_name + '/' + file)
        # Save spectrum
        model = {}
        model['Name'] = f'T{T * 1000}logg{log_g / 10}'
        model['WAVELENGTH'] = spectrum['WAVELENGTH']
        model['FLUX'] = spectrum['FLUX']
        model['Teff'] = T * 1000
        model['log(g)'] = log_g / 10

        models[f'T{T * 1000}logg{log_g / 10}'] = model

    # Save the models as json file
    with open(folder_name + '_save.json', 'w') as json_file:
        json.dump(models, json_file)

    return models



def import_models_quickload(galaxy:str)->dict:
    """
    Imports all models for a given galaxy

    Args:
        galaxy (str): Galaxy to import the models for (Milkyway, SMC, LMC)

    Returns:
        models (dict): Dictionary with models. Name: T{Teff}logg{log(g)}
    """
    
    # Path to the models
    model_path = '/mnt/c/Users/luukv/Documenten/NatuurSterrkenkundeMasterProject/CodeMP/MasterProject/ModelFitting/Models/'

    # Assign folder name corresponding to the galaxy
    if galaxy == 'Milkyway':
        folder_name = model_path + 'gal-ob-i_line_all'
    elif galaxy == 'SMC':
        folder_name = model_path + 'smc-ob-i_line_all'
    elif galaxy == 'LMC':
        folder_name = model_path + 'lmc-ob-i_line_all'

    # Load the JSON file with the models
    with open(folder_name + '_save.json', 'r') as json_file:
        models = json.load(json_file)

    return models



def divide_colormap(num_parts: int, colormap_name: str='rainbow')->list:
    """
    Devides a color map in the given number of parts

    Args:
        num_parts (int): Number of colors you want returned
        colormap_name (str, optional): Name of the colormap. Defaults to 'rainbow'.

    Returns:
        list: list of color names
    """
    cmap = plt.get_cmap(colormap_name)
    num_colors = cmap.N
    colors = [cmap(i / num_colors) for i in range(0, num_colors, num_colors // num_parts)]
    return colors



def plot_models_and_spectrum(models: dict, spectra: dict, object_name: str, distance: float, spectral_lines: bool=False)->None:
    """
    Plot the given models and spectra

    Args:
        models (dict): Models to be plotted
        spectra (dict): Spectra to be plotted
        object_name (str): Name of the object of the spectra
        spectral_lines (bool): If True, spectral lines will be shown in plot
    """

    # Make a list with different colors
    colors = divide_colormap(len(models))

    plt.figure(figsize=(12,8))
    
    # Plot the spectra
    for spectrum in spectra:
        plt.plot(spectrum[0], spectrum[1], color='midnightblue', label=f'Spectrum {object_name}')

    # Plot the models
    for i, model in enumerate(models.values()):
        plt.plot(model['WAVELENGTH'], model['FLUX'], 
                 color=colors[i], label='Teff {} log(g) {}'.format(model['Teff'], model['log(g)']))

    # Plot spectral lines
    if spectral_lines:
        # Plot the spectral lines
        colors = divide_colormap(len(important_spectral_lines()))
        for i, line in enumerate(important_spectral_lines()):
            plt.vlines(line[0], ymin=0, ymax=1, label=line[1], color=colors[i])

    plt.title('Spectrum and models', size=18)
    plt.ylabel(r'log $F_{\lambda}$ / (erg cm$^{-2}$ s$^{-1}$ $\AA^{-1}$)', fontsize= 12)
    plt.xlabel(r'$\lambda$ ($\AA$)', fontsize=12)
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.show()

    return


def models_in_interval(models: dict, T1: int, T2: int, log_g1: float, log_g2: float)->dict:
    """
    Gives back a dictionary with only the models in the given T,
    log(g) range.

    Args:
        models (dict): Dictionary of all models
        T1 (int): Start T interval
        T2 (int): End T interval
        log_g1 (float): Start log(g) interval
        log_g2 (float): End log(g) interval

    Returns:
        dict: All models in the given T,log(g) range
    """

    # Get all Teff, log(g) combinations
    available_Teff_logg = []
    for model in models.values():
        available_Teff_logg.append((model['Teff'], model['log(g)']))

    # Filter only models in the given interval
    filtered_Teff_logg = [f'T{Teff}logg{log_g}' for (Teff, log_g) in available_Teff_logg if T1 <= Teff <= T2 and log_g1 <= log_g <= log_g2]

    # Make dictionary with models in given interval
    filtered_models = {key: models[key] for key in filtered_Teff_logg if key in models}

    return filtered_models



def important_spectral_lines()->list:
    """
    List of important spectral lines with their name and wavelengths

    Returns:
        list: list of spectral lines
    """

    spectral_lines: list = [
        (4026, "He I + II 4026"),
        (4200, "He II 4200"),
        (4634, "N III 4634-40-42 (emission)"),
        (4686, "He II 4686"),
        (4144, "He I: 4144"),
        (4388, "He I: 4388"),
        (4541, "He II 4541"),
        (4552, "Si II 4552"),
        (4686, "He II 4686"),
        (4861, r"H$\beta$: 4861"),
        (5016, "He I: 5016"),
        (5876, "He I: 5876"),
        (5890, "Na I 5890"),
        (5896, "Na I 5896"),
        (6527, "He II 6527"),
        (6563, "Ha 6563"),
        (4471, "He I: 4471"),
        (4058, "N IV 4058"),
        (4116, "Si IV 4116"),
        (4097, "N III 4097"),
        (4504, "Si IV 4686-4504"),
        (4713, "He I: 4713"),
        (4187, "C III 4187"),
        (4121, "He I: 4121"),
        (3995, "N II 3995"),
        (4350, "O II 4350"),
        (4128, "Si I 4128-30"),
        (4481, "Mg II 4481"),
        (4233, "Fe II 4233"),
    ]

    return spectral_lines



def vandermeer_lines(element:str =None)->list:
    """
    Returns a list of elements reported in van der Meer (2007) table 3.

    Args:
        element (str, optional): Choose [balmer, helium1, other]. Defaults to None.

    Returns:
        list: List of tuples (wavelength, line name)
    """
    balmer = [
        (4861.33, r"H$\beta$: 4861.33"),
        (4340.46, r"H$\gamma$: 4340.46"),
        (4101.73, r"H$\delta$: 4101.73"),
        (3970.07, r"H$\epsilon$: 3970.07"),
        (3889.05, r"H8: 3889.05"),
        (3835.38, r"H9: 3835.38"), 
        (3797.90, r"H10: 3797.90"),
        (3770.63, r"H11: 3770.63"),
        (3750.15, r"H12: 3750.15"),
        (3734.37, r"H13: 3734.37"),
        (3721.94, r"H14: 3721.94"),
        (3711.97, r"H15: 3711.97"),
        (3703.85, r"H16: 3703.85")
    ]

    helium1 = [
        (5875.66, r"He I: 5875.66"),
        (4471.50, r"He I: 4471.50"),
        (4026.21, r"He I: 4026.21"),
        (3819.62, r"He I: 3819.62"),
        (3705.02, r"He I: 3705.02"),
        (3634.25, r"He I: 3634.25"),
        (3587.27, r"He I: 3587.27"),
        (4921.93, r"He I: 4921.93"),
        (4387.93, r"He I: 4387.93"),
        (4143.76, r"He I: 4143.76"),
        (4009.26, r"He I: 4009.26"),
        (4713.17, r"He I: 4713.17"),
        (5015.68, r"He I: 5015.68"),
        (5047.74, r"He I: 5047.74")
    ]

    other = [
        (5411.53, r"He II: 5411.53"),
        (4199.83, r"He II: 4199.83"),
        (4088.86, r"Si IV: 4088.86"),
    ]

    if element == "balmer":
        return balmer
    if element == "Helium1":
        return helium1
    if element == "other":
        return other
    if element == None:
        return balmer + helium1 + other



def extract_temperature_and_gravity(model_name):
    # Define the regular expression pattern
    pattern = r"T(\d+)logg([\d.]+)"
    
    # Match the pattern in the model name
    match = re.match(pattern, model_name)
    
    if match:
        temperature = int(match.group(1))
        gravity = float(match.group(2))
        return temperature, gravity
    else:
        return None, None




"""############################################################################################
ALL FUNCTIONS BELOW ARE USED IN THE GRID SEARCH
"""############################################################################################



def select_spectrum(spectra: list, central_wav: float)->tuple:
    """
    Select the spectrum containing the given spectral line

    Args:
        spectra (list): List of spectra, made by import_spectra(object)
        central_wav (float): Central wavelenght of the spectral line

    Returns:
        tuple: wavelength, flux
    """
    for spectrum in spectra[::-1]:
        if central_wav >= min(spectrum[0])  and central_wav <= max(spectrum[0]):
            return spectrum[0], spectrum[1]



def extract_continuum(wavelengths: np.array, flux: np.array, start: float, end: float, line_left: float, line_right: float)->tuple:
    """
    Extract continuum from spectrum

    Args:
        wavelengths (np.array): List with wavelength
        flux (np.array): List with flux
        start (float): Start of spectrum
        end (float): End of spectrum
        line_left (float): Start spectral line
        line_right (float): End spectral line

    Returns:
        tuple: (Wavelength, Flux)
    """
    # Find indices corresponding to the specified wavelength range
    indices_left = np.where((wavelengths >= start) & (wavelengths <= line_left))[0]
    indices_right = np.where((wavelengths >= line_right) & (wavelengths <= end))[0]
    indices = np.concatenate((indices_left, indices_right))

    # Extract wavelength and flux values within the range
    extracted_wavelengths = wavelengths[indices]
    extracted_flux = flux[indices]

    return extracted_wavelengths, extracted_flux



def chi_squared(wav_model, flux_model, wav_line, flux_line, SNR):

    # Get the flux for every wavelenght of the data.
    # Create a CubicSpline object
    cubic_spline = CubicSpline(wav_model, flux_model)

    # Interpolate intensity at the desired wavelengths
    flux_model_inter = cubic_spline(wav_line)

    # Calculate chi-squared
    chi_squared = 0
    for i in range(len(flux_line)):
        chi_squared += ( (flux_model_inter[i] - flux_line[i]) / (1 / SNR) ) ** 2
    chi_squared /= len(wav_line)

    return chi_squared



def gaussian(x: list, mean: float, amplitude: float, stddev: float, continuum: float)->list:
    """
    Gauss

    Args:
        x (list): Data
        mean (float): mu
        amplitude (float): amplitude
        stddev (float): sigma
        continuum (float): c parameter

    Returns:
        list: y-values of gauss
    """
    return amplitude * np.exp(-((x - mean) / stddev) ** 2 / 2) + continuum



def determine_doppler_shift(spectra: list, lines: dict, guassian: callable, object_name:str, plot: bool=False, save=False)->list:
    """
    This function takes a spectra and list with lines (and their ranges)
    and fits a gaussian to these lines to determine the doppler shift of
    the spectrum.

    Args:
        spectra (list): Spectra of the object (UVES)
        lines (dict): Spectral lines and their ranges
        guassian (callable): Function of a gaussian
        plot (bool, optional): If true, the fits to the data are shown. Defaults to False.
        save (_type_): If a savepath is given the plots are being saved

    Returns:
        list: All doppler shift of the individual lines.
    """
    ## FIT ALL LINES
    # Dictionary to save the fit results.
    fit_results = {}

    # Fit a gauss to all spectral lines
    for line in lines['lines']:
        # Rest wavelength of the spectral line
        central_wavelength = line[0]

        # Select the spectrum that contains the spectral line
        wav, flux = select_spectrum(spectra, central_wavelength)

        # Extract the spectral line from the spectrum
        wav, flux = extract_spectrum_within_range(wav, flux, line[2], line[5])
        wav_cont, flux_cont = extract_continuum(wav, flux, line[2], line[5], line[3], line[4])
        wav_line, flux_line = extract_spectrum_within_range(wav, flux, line[3], line[4])

        # Initial guess for the parameters
        initial_guess = [line[0] - lines['Doppler_guess'], # mu
                         max(flux_line) - np.mean(flux_cont), # amplitude
                        (max(wav_line) - min(wav_line)) / 3.5, # stddev
                        np.mean(flux_cont)] # continuum height

        # Fit the data
        params, covariance = curve_fit(gaussian, wav, flux, p0=initial_guess)

        # Save results
        fit_results[line[1]] = {"spectrum": (wav, flux),
                                "continuum": (wav_cont, flux_cont),
                                "line": (wav_line, flux_line),
                                "fit_result": (params, covariance),
                                "rest_wavelength": line[0]}


    ## DETERMINE THE DOPPLER SHIFT
    # Calculate the dopplershift for every line
    doppler_shift = []
    for line, data in fit_results.items():
        lambda0 = data['rest_wavelength']
        delta_lambda = data['fit_result'][0][0] - lambda0
        velocity = delta_lambda / lambda0 * 3E5 # km/s
        doppler_shift.append(velocity)

    print(f'The velocity of the object is: {np.mean(doppler_shift)} +- {np.std(doppler_shift)}')


    ## PLOT THE FIT RESULTS
    if plot:
        num_plots = len(fit_results)
        num_rows = (num_plots - 1) // 4 + 1  # Calculate the number of rows needed

        fig, axes = plt.subplots(num_rows, 4, figsize=(15, num_rows * 4))

        # Flatten axes if necessary
        if num_rows == 1:
            axes = [axes]

        for i, ax_row in enumerate(axes):

            for j, ax in enumerate(ax_row):
                plot_index = i * 4 + j

                if plot_index < num_plots:
                    key = list(fit_results.keys())[plot_index]
                    wav, flux = fit_results[key]['spectrum']
                    fit = gaussian(wav, *fit_results[key]['fit_result'][0])

                    ax.plot(wav, flux, color='blue', label='spectrum')  # Plot your data here
                    ax.plot(wav, fit, color='orange', label='fit')
                    ax.vlines(fit_results[key]['fit_result'][0][0], ymin=min(fit), ymax=max(fit),
                            label=(r'$\mu$ = ' + f"{round(fit_results[key]['fit_result'][0][0], 2)}" + r'$\AA$'), color='red')
                    ax.set_title(f'{key}')
                    ax.legend(fontsize=8)

                else:
                    ax.axis('off')  # Turn off axis for unused subplots
                
                if j == 0:
                    ax.set_ylabel('flux', size=12)
                if i == len(axes) - 1:
                    ax.set_xlabel(r'Wavelength ($\AA$)', size=12)


        plt.suptitle(f"{object_name}\nRadial velocity: {round(np.mean(doppler_shift), 2)}" + r" $\pm$ " + f"{round(np.std(doppler_shift), 2)}" + r" km $s^{-1}$", size=20)
        plt.tight_layout()
        plt.savefig(save)
        plt.show()

    return doppler_shift



def doppler_shift_spectrum(wavelengths:Iterable[float], vrad:float)->Iterable[float]:
    """
    Doppler shifts the wavelenghts for the given radial velocity

    Args:
        wavelengths (Iterable[float]): Wavelenghts
        vrad (float): Radial velocity

    Returns:
        Iterable[float]: Doppler shifted wavelengths
    """
    return [(i * (vrad / 299792.458 + 1)) for i in wavelengths]



def chi_squared_for_all_models(spectra:list, models:dict, lines:dict, SNR:list, vrad: float, vsini: float)->dict:
    """
    Determines the chi-squared for every model for the given spectral lines.

    Args:
        spectra (list): The spectra of the object (UVES)
        models (dict): Dictionary with the models
        lines (dict): Dictionary with the lines and their ranges
        SNR (list): Signal to noise ratio of the spectrum
        vrad (float): The doppler_shift of the object in km/s
        vsini (float): the vsini parameter of the object in km/s

    Returns:
        dict: Dictionary with the chi-squared for every model
    """
    # Number of spectral lines
    N_lines = len(lines['lines'])

    # Set the dictionaries to save the chi-squared value
    chi2 = {}
    chi2_perline = {}

    for key in models.keys():
        # Chi-squared parameter
        chi2[key] = 0
        # Chi-squared per line
        chi2_perline[key] = {}

    # Fit the models to every given spectral line
    for i, line in enumerate(lines['lines']):
        print(f"\r\t\tLine {i+1} out of {N_lines}: {line[1]}", end='', flush=True)
        # Rest wavelength of the spectral line
        central_wavelength = line[0]

        # Select the spectrum that contains the spectral line
        wav, flux = select_spectrum(spectra, central_wavelength)
        SNR_value = select_SNR(wav, SNR)

        # Extract the spectral line from the spectrum
        wav, flux = extract_spectrum_within_range(wav, flux, line[2], line[5])
        wav_cont, flux_cont = extract_continuum(wav, flux, line[2], line[5], line[3], line[4])
        wav_line, flux_line = extract_spectrum_within_range(wav, flux, line[3], line[4])

        # Linear fit to continuum
        cont_fit = np.poly1d(np.polyfit(wav_cont, flux_cont, 1))
        # Normalize spectrum
        flux /= cont_fit(wav)
        flux_cont /= cont_fit(wav_cont)
        flux_line /= cont_fit(wav_line)

        # Fit all models to the spectral line
        for key, model in models.items():

            # Extract the line from the model
            wav_model, flux_model = extract_spectrum_within_range(np.array(model['WAVELENGTH']), np.array(model['FLUX']), line[2], line[5])
            # Dopplershift the model
            wav_model = doppler_shift_spectrum(wav_model, vrad)

            # Apply doppler broadening
            wav_model, flux_model = pyasl.equidistantInterpolation(wav_model, flux_model, "2x")
            flux_model = pyasl.rotBroad(wav_model, flux_model, 0.0, vsini)

            # Calculate chi-squared for this line
            chi2_value = chi_squared(wav_model, flux_model, wav_line, flux_line, SNR_value)
            # Keep track of the total chi-squared for all lines
            chi2[key] += chi2_value
            # Save for each line the chi-squared individually
            chi2_perline[key][line[1]] = chi2_value

    # Normalize all chi-squared values
    for key in models.keys():
        # Devide the total chi-squared by the number of lines.
        chi2[key] /= len(lines['lines'])

    print("DONE", end='', flush=True)
    return chi2, chi2_perline



def plot_best_model(spectra: list, models:dict, lines:dict, best_model:str, vrad:float, vsini:float, save=False)->None:
    """
    Plots the best model over the spectrum

    Args:
        spectra (list): Spectrum of the object (UVES)
        models (dict): All models
        lines (dict): List with all lines and their ranges
        best_model (str): Model with the lowest chi-squared
        vrad (float): Radial velocity (km/s)
        vsini (float): vsin(i) (km/s)
    """
    # Make subplots
    num_plots = len(lines['lines'])
    num_rows = (num_plots - 1) // 4 + 1  # Calculate the number of rows needed
    fig, axes = plt.subplots(num_rows, 4, figsize=(15, num_rows * 4))
    x = 0
    y = 0

    # Flatten axes if necessary
    if num_rows == 1:
        axes = [axes]

    model = models[best_model]
    for line in lines['lines']:

        # Rest wavelength of the spectral line
        central_wavelength = line[0]

        # Select the spectrum that contains the spectral line
        wav, flux = select_spectrum(spectra, central_wavelength)

        # Extract the spectral line from the spectrum
        wav, flux = extract_spectrum_within_range(wav, flux, line[2], line[5])
        wav_cont, flux_cont = extract_continuum(wav, flux, line[2], line[5], line[3], line[4])
        wav_line, flux_line = extract_spectrum_within_range(wav, flux, line[3], line[4])
        # Extract the line from the model
        wav_model, flux_model = extract_spectrum_within_range(np.array(model['WAVELENGTH']), np.array(model['FLUX']), line[2], line[5])

        # Linear fit to continuum
        cont_fit = np.poly1d(np.polyfit(wav_cont, flux_cont, 1))
        # Normalize spectrum
        flux /= cont_fit(wav)
        flux_cont /= cont_fit(wav_cont)
        flux_line /= cont_fit(wav_line)

        # Dopplershift the model
        wav_model = doppler_shift_spectrum(wav_model, vrad)

        # Apply doppler broadening
        wav_model, flux_model = pyasl.equidistantInterpolation(wav_model, flux_model, "2x")
        flux_model = pyasl.rotBroad(wav_model, flux_model, 0.0, vsini)

        axes[x,y].plot(wav, flux, color='blue', alpha=0.5)
        axes[x,y].plot(wav_line, flux_line, color='orange', alpha=0.5)
        axes[x,y].plot(wav_model, flux_model, color='green')

        # Annotate each line with text vertically
        axes[x,y].set_title(line[1], fontsize=10)
        axes[x,y].grid(alpha=0.25)

        if y == 0:
            axes[x,y].set_ylabel('Normalised Flux', fontsize=12)
        if x == num_rows - 1:
            axes[x,y].set_xlabel(r"$\lambda$ ($\AA$)", fontsize=12)

        # Set right plot coordinates
        if (y + 1) % 4 == 0:
            x += 1
            y = 0
        else:
            y += 1

    plt.suptitle(f'Best model of the lines\n{best_model}', fontsize=15)
    plt.tight_layout()
    if save:
        plt.savefig(save)
    plt.show()

    return



def plot_continuum_fits(spectra, lines):

    # Lists to save the data
    wav_list = []
    flux_list = []
    fit_list = []
    wav_line_list = []
    flux_line_list = []

    # A linear continuum fit for every spectral line
    for line in lines['lines']:
        # Rest wavelength of the spectral line
        central_wavelength = line[0]

        # Select the spectrum that contains the spectral line
        wav, flux = select_spectrum(spectra, central_wavelength)

        # Extract the spectral line from the spectrum
        wav, flux = extract_spectrum_within_range(wav, flux, line[2], line[5])
        wav_cont, flux_cont = extract_continuum(wav, flux, line[2], line[5], line[3], line[4])
        wav_line, flux_line = extract_spectrum_within_range(wav, flux, line[3], line[4])

        # Linear fit to continuum
        cont_fit = np.poly1d(np.polyfit(wav_cont, flux_cont, 1))(wav_cont)

        wav_list.append(wav_cont)
        flux_list.append(flux_cont)
        fit_list.append(cont_fit)
        wav_line_list.append(wav_line)
        flux_line_list.append(flux_line)


    num_plots = len(lines['lines'])
    num_rows = (num_plots - 1) // 4 + 1  # Calculate the number of rows needed

    fig, axes = plt.subplots(num_rows, 4, figsize=(15, num_rows * 4))

    # Flatten axes if necessary
    if num_rows == 1:
        axes = [axes]

    for i, ax_row in enumerate(axes):

        for j, ax in enumerate(ax_row):
            plot_index = i * 4 + j

            if plot_index < num_plots:

                ax.plot(wav_list[i], flux_list[i], color='blue', label='continuum')  # Plot your data here
                ax.plot(wav_list[i], fit_list[i], color='orange', label='fit')
                ax.plot(wav_line_list[i], flux_line_list[i], color='green', label='line')
                ax.set_title(f"{lines['lines'][i][1]}")
                ax.legend(fontsize=8)

            else:
                ax.axis('off')  # Turn off axis for unused subplots
            
            if j == 0:
                ax.set_ylabel('flux', size=12)
            if i == len(axes) - 1:
                ax.set_xlabel(r'Wavelength ($\AA$)', size=12)

    plt.tight_layout()
    plt.show()

    return



def plot_models_over_lines(spectra:list, models:dict, lines:dict, vrad:float, vsini:float):
    """
    Plot the given models over the spectral lines.

    Args:
        spectra (list): _description_
        models (dict): _description_
        lines (dict): _description_
        vrad (float): _description_
        vsini (float): _description_
    """
    # Make subplots
    num_plots = len(lines['lines'])
    num_rows = (num_plots - 1) // 4 + 1  # Calculate the number of rows needed
    fig, axes = plt.subplots(num_rows, 4, figsize=(15, num_rows * 4))
    x = 0
    y = 0

    # Flatten axes if necessary
    if num_rows == 1:
        axes = [axes]

    for line in lines['lines']:

        # Rest wavelength of the spectral line
        central_wavelength = line[0]

        # Select the spectrum that contains the spectral line
        wav, flux = select_spectrum(spectra, central_wavelength)

        # Extract the spectral line from the spectrum
        wav, flux = extract_spectrum_within_range(wav, flux, line[2], line[5])
        wav_cont, flux_cont = extract_continuum(wav, flux, line[2], line[5], line[3], line[4])
        wav_line, flux_line = extract_spectrum_within_range(wav, flux, line[3], line[4])

        # Linear fit to continuum
        cont_fit = np.poly1d(np.polyfit(wav_cont, flux_cont, 1))
        # Normalize spectrum
        flux /= cont_fit(wav)
        flux_cont /= cont_fit(wav_cont)
        flux_line /= cont_fit(wav_line)

        for key, model in models.items():
            # Extract the line from the model
            wav_model, flux_model = extract_spectrum_within_range(np.array(model['WAVELENGTH']), np.array(model['FLUX']), line[2], line[5])

            # Dopplershift the model
            wav_model = doppler_shift_spectrum(wav_model, vrad)

            # Apply doppler broadening
            wav_model, flux_model = pyasl.equidistantInterpolation(wav_model, flux_model, "2x")
            flux_model = pyasl.rotBroad(wav_model, flux_model, 0.0, vsini)

            # Plot the model
            axes[x,y].plot(wav_model, flux_model, label=f'{key}')

        axes[x,y].plot(wav, flux, color='blue', alpha=0.5)
        axes[x,y].plot(wav_line, flux_line, color='orange', alpha=0.5)


        # Annotate each line with text vertically
        axes[x,y].set_title(line[1], fontsize=10)
        axes[x,y].grid(alpha=0.25)

        if y == 0:
            axes[x,y].set_ylabel('Normalised Flux', fontsize=12)
        if x == num_rows - 1:
            axes[x,y].set_xlabel(r"$\lambda$ ($\AA$)", fontsize=12)

        # Set right plot coordinates
        if (y + 1) % 4 == 0:
            x += 1
            y = 0
        else:
            y += 1

    plt.suptitle(f'All models over the lines', fontsize=15)
    plt.tight_layout()
    plt.show()

    return



def lines(object_name:str)->dict:
    """
    Returns a dictionary with the spectral lines and their ranges that
    will be used in the model fitting.

    Args:
        object_name (str): Name of the object you want the lines for.

    Returns:
        dict: lines: [wavelength, name, continuum left, line left, line right, continuum right]
              doppler_guess: #
    """
    # List with the lines that are appropriate to fit a gauss
    # lines: [wavelength, name, continuum left, line left, line right, continuum right]
    _4U1538_52 = {
        'lines': [
            [4340.46, r"H$\gamma$: 4340.46", 4325, 4335, 4340.7, 4347],
            [4387.93, r"He I: 4387.93", 4381, 4383.75, 4388.20, 4393],
            [4471.50, r"He I: 4471.50", 4459, 4466.2, 4472, 4477.5],
            [4713.17, r"He I: 4713.17", 4703, 4708.5, 4713, 4717],
            [4861.33, r"H$\beta$: 4861.33", 4845, 4854.5, 4862, 4870], 
            [4921.93, r"He I: 4921.93", 4910, 4916, 4922.4, 4932],
            [5015.68, r"He I: 5015.68", 5000, 5010.4, 5016, 5024],
        ],
        'Doppler_guess': 2.8
    }

    _CenX_3 = {
        'lines': [
            [4861.33, r"H$\beta$: 4861.33", 4846, 4855, 4867.5, 4874.5],
            [4921.93, r"He I: 4921.93", 4907, 4917.8, 4926.2, 4934],
            [5015.68, r"He I: 5015.68", 5000, 5011, 5020.5, 5030],
            [5411.53, r"He II: 5411.53", 5405.75, 5407, 5417.5, 5423],
            [5875.66, r"He I: 5875.66", 5859, 5870, 5882, 5887.7],
            [4199.83, r"He II: 4199.83", 4191, 4196.8, 4204, 4210],
            [4340.46, r"H$\gamma$: 4340.46", 4326, 4335.6, 4345.4, 4355],
            [3770.63, r"H11: 3770.63", 3764, 3766.5, 3774.7, 3778],
            [3797.90, r"H10: 3797.90", 3791, 3794.25, 3801.5, 3804.5],
            [3819.62, r"He I: 3819.62", 3812.7, 3816.7, 3822.3, 3826],
            [3835.38, r"H9: 3835.38", 3827, 3831, 3839.25, 3845],
            [3889.05, r"H8: 3889.05", 3875, 3884, 3894, 3902]
        ],
        'Doppler_guess': 0
    }

    _SMCX_1 = {
        'lines': [
            [4713.17, r"He I: 4713.17", 4710, 4713.3, 4719.7, 4724],
            [4861.33, r"H$\beta$: 4861.33", 4845, 4860.5, 4869, 4879],
            [4921.93, r"He I: 4921.93", 4915, 4921.3, 4928.9, 4934],
            [5015.68, r"He I: 5015.68", 5010, 5015.5, 5022.5, 5029],
            [5047.74, r"He I: 5047.74", 5043, 5047.8, 5054.5, 5061]
            # [5411.53, r"He II: 5411.53", 5400, 5408.5, 5418.5, 5424]
            #[5875.66, r"He I: 5875.66", ]
            # [3587.27, r"He I: 3587.27", 3585.5, 3586.7, 3591.8, 3594.2],
            #[4009.26, r"He I: 4009.26", 4005, 4009.25, 4014.8, 4018],
            #[4026.21, r"He I: 4026.21", 4024, 4025.75, 4031.5, 4035]
        ],
        'Doppler_guess': -3.5
    }

    _4U1700_37 = {
        'lines': [
            # [3835.38, r"H9: 3835.38", 3830, 3832, 3837.5, 3840],
            [3889.05, r"H8: 3889.05", 3882.75, 3884.5, 3891.75, 3896],
            [4026.21, r"He I: 4026.21", 4019.5, 4021.5, 4027.75, 4030.5],
            [4058, r"N IV 4058", 4053, 4055, 4059.7, 4062],
            [4199.83, r"He II: 4199.83", 4195.5, 4196.6, 4201.9, 4204],
            [4541, r"He II 4541", 4535.4, 4536.4, 4544.25, 4545]
        ],
        'Doppler_guess': 0.6
    }

    _LMCX_4 = {
        'lines': [
            [4861.33, r"H$\beta$: 4861.33", 4848, 4859, 4873, 4880],
            [4921.93, r"He I: 4921.93", 4910, 4922.1, 4932.2, 4946],
            [5411.53, r"He II: 5411.53", 5405, 5412.5, 5422.3, 5427],
            [4340.46, r"H$\gamma$: 4340.46", 4335, 4339.6, 4350, 4354],
            [4471.50, r"He I: 4471.50", 4467, 4472.3, 4480.5, 4485]
        ],
        'Doppler_guess': -5
    }

    _VelaX_1 = {
        'lines': [
            [4009.26, r"He I: 4009.26", 4004, 4006.3, 4011.25, 4013],
            [4143.76, r"He I: 4143.76", 4139, 4141.5, 4146, 4148],
            [4088.86, r"Si IV: 4088.86", 4085, 4086.2, 4091, 4093],
            [4199.83, r"He II: 4199.83", 4197.55, 4198, 4201.6, 4203],
            [4387.93, r"He I: 4387.93", 4383.8, 4385, 4390.5, 4392.3],
            [4921.93, r"He I: 4921.93", 4916, 4918.5, 4926, 4932],
            [5411.53, r"He II: 5411.53", 5406.8, 5409, 5413.75, 5416.5]
        ],
        'Doppler_guess': 0.3
    }

    line_dict = {'4U1538-52': _4U1538_52, 'Cen X-3': _CenX_3, 'SMC X-1': _SMCX_1,
                 '4U1700-37': _4U1700_37, 'LMC X-4': _LMCX_4, 'Vela X-1': _VelaX_1}

    return line_dict[object_name]



def select_SNR(wav:list, SNR:list)->float:
    """
    Select the SNR that corresponds to the spectrum

    Args:
        wav (list): Wavelength of the spectrum
        SNR (list): List with SNR of all spectra

    Returns:
        float: SNR
    """
    for i in SNR:
        if i[0] >= min(wav) and i[0] <= max(wav):
            return i[1]



def SignalToNoise(object_name:str)->list:
    """
    Returns the SNR of the spectra of the given object

    Args:
        object_name (str): Name of the object

    Returns:
        (list): List with SNR of the objects spectra
    """
    SNR_4U1538_52 = [(4000, 9.), (5000, 63.7)]
    SNR_CenX_3 = [(4000, 32.8), (5000, 90.7)]
    SNR_SMCX_1 = [(4000, 50.2), (5000, 69.8)]
    SNR_4U1700_37 = [(4000, 390.1), (7000, 310.3)]
    SNR_LMCX_4 = [(4000, 59.5), (5000, 64.8)]
    SNR_VelaX_1 = [(4000, 378.9)]

    SNR = {'4U1538-52': SNR_4U1538_52, 'Cen X-3': SNR_CenX_3, 'SMC X-1': SNR_SMCX_1,
           '4U1700-37': SNR_4U1700_37, 'LMC X-4': SNR_LMCX_4, 'Vela X-1': SNR_VelaX_1}

    return SNR[object_name]



def extract_vsini(vsini:str)->int:
    """
    Takes a string like 'vsini###'
    This function extracts the number

    Args:
        vsini (str): String like 'vsini###'

    Returns:
        int: The number in the string
    """
    return int(re.search(r'\d+$', vsini).group())