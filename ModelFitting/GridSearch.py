import os
import json
import pandas as pd
import sys
sys.path.append('/mnt/c/Users/luukv/Documenten/NatuurSterrkenkundeMasterProject/CodeMP/MasterProject')

from model_functions import *
from functions import import_spectra, extract_spectrum_within_range


"""
SET INITIAL PARAMETERS
"""
# SET THE NAME OF THE OBJECT YOU WANT TO DO THE GRID SEARCH FOR
OBJECT_NAME = '4U1700-37'
object_ = '4U1700_37'

# Set the galaxy for the models
galaxy = 'Milkyway'

# Grid Search values for vsin(i)
vsini_start = 100
vsini_end = 250
vsini_stepsize = 5


"""
TEST IF DIRECTORY IS REACHABLE
"""
print("MAKE NEW DIRECTORY")
# Folder path to FitResults folder
folder_path = '/mnt/c/Users/luukv/Documenten/NatuurSterrkenkundeMasterProject/CodeMP/MasterProject/ModelFitting/FitResults/'
# Make a new folder to save the results
new_folder_name = f'{OBJECT_NAME}S{vsini_start}E{vsini_end}delta{vsini_stepsize}' # You can specify any path here
# Create the new folder
os.makedirs(folder_path + new_folder_name, exist_ok=True)

# Make dummy JSON file
with open(folder_path + new_folder_name + '/' + 'test.json', 'w') as json_file:
    json.dump([], json_file)
# Load dummy JSON file
with open(folder_path + new_folder_name + '/' + 'test.json', 'r') as json_file:
    loaded_data = json.load(json_file)
# Remove the dummy file
os.remove(folder_path + new_folder_name + '/' + 'test.json')
print('\tDIRECTORY FOUND')


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

# Signal-to-noise ratios
SNR = SignalToNoise(OBJECT_NAME)


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
print("\nSTART GRID SEARCH")
# Make results dictionary
results = {}
results_perline = {}

# Set the grid for vsin(i)
vsini_grid = list(range(vsini_start, vsini_end + 1, vsini_stepsize))

# PERFORM THE GRID SEARCH
for vsini in vsini_grid:
    print(f"\tvsin(i): {vsini}", flush=True)
    chi2, chi2_perline = chi_squared_for_all_models(spectra, models, _object_lines, SNR, vrad, vsini)
    results[f'vsini{vsini}'] = chi2
    results_perline[f'vsini{vsini}'] = chi2_perline
    print()


"""
SAVE RESULTS
"""
# Save result as json file
with open(folder_path + new_folder_name + '/' + 'Chi2.json', 'w') as json_file:
    json.dump(results, json_file)
# Save result per line as json file
with open(folder_path + new_folder_name + '/' + 'Chi2_perline.json', 'w') as json_file:
    json.dump(results_perline, json_file)


# Save result as DataFrame
pd.DataFrame.from_dict(results, orient='columns').to_csv(folder_path + new_folder_name + '/' + 'Chi2.csv')
# Save result per line as DataFrame
pd.DataFrame.from_dict(results_perline, orient='columns').to_csv(folder_path + new_folder_name + '/' + 'Chi2_perline.csv')

print("\nDONE")