import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re

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