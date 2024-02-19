import json
import pandas as pd
import sys
sys.path.append('/mnt/c/Users/luukv/Documenten/NatuurSterrkenkundeMasterProject/CodeMP/MasterProject')

from model_functions import *
from functions import import_spectra, extract_spectrum_within_range


"""
TEST IF DIRECTORY IS REACHABLE
"""
folder_path = '/mnt/c/Users/luukv/Documenten/NatuurSterrkenkundeMasterProject/CodeMP/MasterProject/ModelFitting/FitResults/'
# Load the JSON file
with open(folder_path + 'test.json', 'r') as json_file:
    loaded_data = json.load(json_file)
print('DIRECTORY FOUND')


"""
SET INITIAL PARAMETERS
"""
# SET THE NAME OF THE OBJECT YOU WANT TO DO THE GRID SEARCH FOR
OBJECT_NAME = '4U1538-52'
object_ = '4U1538_52'

# Set the galaxy for the models
galaxy = 'Milkyway'

# Signal to noise ration of the spectra
SNR = 9

# Grid Search values for vsin(i)
vsini_start = 100
vsini_end = 250
vsini_stepsize = 5


"""
Import all necesities
"""
# Import spectrum of given object
spectra = import_spectra(object_)
print('SPECTRA IMPORTED')

# Import models for given galaxy
models = import_models(galaxy)
print('MODELS IMPORTED')

# The spectral lines that correspond to the object and are used
# in the model fitting
_object_lines = lines(OBJECT_NAME)


"""
RADIAL VELOCITY
"""
# Determine the radial velocity of the object
doppler_shifts = determine_doppler_shift(spectra, _object_lines, gaussian, False)
vrad = np.mean(doppler_shifts)
vrad_err = np.std(doppler_shifts)


"""
GRID SEARCH
"""
# Make results dictionary
results = {}

# Set the grid for vsin(i)
vsini_grid = list(range(vsini_start, vsini_end + 1, vsini_stepsize))

# PERFORM THE GRID SEARCH
for vsini in vsini_grid:
    print(f"\nvsin(i): {vsini}", flush=True)
    chi2 = chi_squared_for_all_models(spectra, models, _object_lines, SNR, vrad, vsini)
    results[f'vsini{vsini}'] = chi2


"""
SAVE RESULTS
"""
# Save result as json file
with open(folder_path + f'{OBJECT_NAME}S{vsini_start}E{vsini_end}delta{vsini_stepsize}.json', 'w') as json_file:
    json.dump(results, json_file)

# Save result as DataFrame
pd.DataFrame.from_dict(results, orient='columns').to_csv(folder_path + f'{OBJECT_NAME}S{vsini_start}E{vsini_end}delta{vsini_stepsize}.csv')