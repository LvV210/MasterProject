import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re

"""
Functions that are used for model fitting
"""

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
        spectrum['FLUX'].append(10 ** (float(values[1])))

    return spectrum



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