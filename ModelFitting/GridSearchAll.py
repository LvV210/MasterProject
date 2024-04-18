import os
import json
import pandas as pd
import sys
sys.path.append('/mnt/c/Users/luukv/Documenten/NatuurSterrkenkundeMasterProject/CodeMP/MasterProject')

from AllModelFunctions import *
from functions import import_spectra


object_names = ['Vela X-1', 'Cen X-3', '4U1538-52', '4U1700-37']
# 'SMC X-1', 'LMC X-4', 
# object_names = ['Vela X-1']
object_saves = ['VelaX_1', 'CenX_3', '4U1538_52', '4U1700_37']
# 'SMCX_1', 'LMCX_4'
# object_saves = ['VelaX_1']
galaxies = ['Milkyway', 'Milkyway', 'Milkyway', 'Milkyway']
# 'SMC', 'LMC', 
# galaxies = ['Milkyway']


for name, save, current_galaxy in zip(object_names, object_saves, galaxies):


    """
    SET INITIAL PARAMETERS
    """
    # SET THE NAME OF THE OBJECT YOU WANT TO DO THE GRID SEARCH FOR
    OBJECT_NAME = name
    object_ = save

    # Set the galaxy for the models
    galaxy = current_galaxy

    # Grid Search values for vsin(i)
    vsini_start = 60
    vsini_end = 260
    vsini_stepsize = 2

    print(f"Current: {OBJECT_NAME} {galaxy}")

    """
    TEST IF DIRECTORY IS REACHABLE
    """
    print("MAKE NEW DIRECTORY")
    # Folder path to FitResults folder
    folder_path = '/mnt/c/Users/luukv/Documenten/NatuurSterrkenkundeMasterProject/CodeMP/MasterProject/ModelFitting/FitResults/'
    # Make a new folder to save the results
    new_folder_name = f'{object_}S{vsini_start}E{vsini_end}delta{vsini_stepsize}' # You can specify any path here
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
    models = import_models_quickload(galaxy)
    print('MODELS IMPORTED')

    # The spectral lines that correspond to the object and are used
    # in the model fitting
    _object_lines = lines_doppler(OBJECT_NAME)

    # Signal-to-noise ratios
    SNR = SignalToNoise(OBJECT_NAME)


    """
    RADIAL VELOCITY
    """
    # Determine the radial velocity of the object
    doppler_shifts = determine_radial_velocity(spectra, _object_lines, gaussian, OBJECT_NAME)
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

    # Get the helium lines for the gridsearch
    _object_lines_He = lines_model_fit(OBJECT_NAME)

    # PERFORM THE GRID SEARCH
    for vsini in vsini_grid:
        print(f"\tvsin(i): {vsini}", flush=True)
        chi2, chi2_perline = chi_squared_for_all_models(spectra, models, _object_lines_He, SNR, vrad, vsini)
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


    """
    WRITE SOME STATISTICS IN A TEST FILE
    """
    # List with all vsin(i) values
    vsini_list = []
    # List with the minimum chi-squared value for every vsin(i)
    min_chi2_list = []
    # List with the best mdel corresponding to the minimum chi-squared for every vsin(i)
    best_models = []

    # Extract the data from the results
    for vsini, chi2 in results.items():
        vsini_list.append(extract_vsini(vsini))
        min_chi2_list.append(chi2[min(chi2, key=chi2.get)])
        best_models.append(min(chi2, key=chi2.get))


    # Best vsin(i)
    vsini_best = list(results.keys())[min_chi2_list.index(min(min_chi2_list))]
    # All chi2 values of the best vsin(i)
    chi2_best = results[vsini_best]
    # Best model of the best vsin(i)
    best_model = min(chi2_best, key=chi2_best.get)


    # Print some first results
    print(f"The minimum chi-squared: \t {round(chi2_best[min(chi2_best, key=chi2_best.get)], 4)}")
    print(f"The best vsin(i): \t\t\t {extract_vsini(vsini_best)} km/s")
    print(f"This is for model: \t\t\t {best_model}")


    # Write some statistics in a text file
    with open(folder_path + new_folder_name + '/' + 'gridsearch_info.txt', 'w') as file:
        # Write some information about the grid search
        file.write("Grid Search Information:\n")
        file.write("------------------------\n")
        file.write(f"Object: \t\t {OBJECT_NAME}\n")
        file.write(f"Object_ : \t\t {object_}\n")
        file.write(f"FolderPath: \t {folder_path + new_folder_name}/\n")
        file.write(f"Galaxy: \t\t {galaxy}\n")
        file.write("\n------------------------\n")
        file.write(f"vsin(i) Start: \t\t {vsini_start}\n")
        file.write(f"vsin(i) End: \t\t {vsini_end}\n")
        file.write(f"vsin(i) Stepsize: \t {vsini_stepsize}\n")
        file.write("\n------------------------\nGRID SEARCH RESULTS\n")
        file.write(f"The minimum chi-squared: \t {chi2_best[min(chi2_best, key=chi2_best.get)]}\n")
        file.write(f"The best vsin(i): \t\t\t {extract_vsini(vsini_best)} km/s\n")
        file.write(f"This is for model: \t\t\t {best_model}\n")
        file.write(f"Radial Velocity: \t\t {vrad}\n")
        file.write(f"Radial Velocity error: \t {vrad_err}\n\n")
        file.write(f"The SNR of the spectra: {SNR}\n")
        file.write(f"The lines examined:\n")
        for line in _object_lines['lines']:
            file.write(f"\t{line[1]}\n")
        file.write(f"\nThe Helium lines used in the grid search:\n")
        for line in _object_lines_He['lines']:
            file.write(f"\t{line[1]}\n")

    print("\nDONE")


    """
    DOING SOME ANALYSIS
    """
    # # Plot the best model over the data
    # plot_best_model(spectra, models, _object_lines_He, best_model, vrad, vsini_best, 
    #             save=(folder_path + new_folder_name + '/' + 'BestModel.png'))
    
    # # Plot all models over the data
    # plot_models_over_lines(spectra, models, _object_lines_He, vrad, vsini_best, 
    #                    (folder_path + new_folder_name + '/' + 'AllModels.png'))
