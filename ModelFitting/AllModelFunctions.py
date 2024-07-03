import numpy as np
from IPython.display import Markdown as md
from tabulate import tabulate
from astropy.constants import R_sun, L_sun, sigma_sb, G, M_sun
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import sympy as sp
from astropy.io import fits
import json
from scipy.optimize import curve_fit
from typing import Iterable
from PyAstronomy import pyasl
from scipy.interpolate import CubicSpline
import re


"""
SOME HANDY FUNCTIONS
"""
def object_name_to_save_name(input_string):
    # Replace every '-' with '_'
    modified_string = input_string.replace('-', '_')
    
    # Remove space before capital 'X'
    modified_string = modified_string.replace(' ', '')

    return modified_string



def save_name_to_object_name(input_string):
    # Replace every '-' with '_'
    modified_string = input_string.replace('_', '-')

    # Place back the spaces after U SMC LMC Vela and XTE
    modified_string = input_string.replace('X-', ' X-')
    modified_string = input_string.replace('XTE', 'XTE ')

    return modified_string



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



def extract_temperature_and_gravity(model_name):
    """
    Takes a string as input like T####logg####
    and returns the 2 numbers.
    """
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



"""
FUNCTIONS TO APPLY ON SPECTRA
"""
def extract_spectrum_within_range(wavelengths: np.array, flux: np.array, start_wavelength: float, end_wavelength: float)->tuple:
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



"""
IMPORT FUNCTIONS
"""
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



def lines_doppler(object_name:str)->dict:
    """
    Returns a dictionary with the spectral lines and their ranges that
    will be used to determine the radial velocity.

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



def lines_model_fit_old(object_name:str, vrad: float)->dict:
    """
    Returns a dictionary with the spectral lines and their ranges that
    will be used in the model fitting.

    Args:
        object_name (str): Name of the object you want the lines for.
        vrad (float): Radial velocity of the system in km/s

    Returns:
        dict: lines: [wavelength, name, continuum left, line left, line right, continuum right]
              doppler_guess: #
    """
    # List with the lines that are appropriate to fit a gauss
    # lines: [wavelength, name, continuum left, line left, line right, continuum right]
    _4U1538_52 = {
        'lines': [
            [4713.17, r"He I: 4713.17", 4708.5, 4709, 4717.35, 4717.85],
            [4009.26, r"He I: 4009.26", 4006, 4006.5, 4012, 4012.5],
            [5875.66, r"He I: 5875.66", 5867, 5867.5, 5882.5, 5883],
            [4026.21, r"He I: 4026.21", 4018.5, 4019, 4031, 4031.5]
        ],
        'Doppler_guess': 2.8
    }

    _CenX_3 = {
        'lines': [
            [4713.17, r"He I: 4713.17", 4708.5, 4709, 4717.35, 4717.85],
            [4009.26, r"He I: 4009.26", 4006, 4006.5, 4012, 4012.5],
            [5875.66, r"He I: 5875.66", 5867, 5867.5, 5882.5, 5883],
            [4026.21, r"He I: 4026.21", 4018.5, 4019, 4031, 4031.5]
        ],
        'Doppler_guess': 0
    }

    _SMCX_1 = {
        'lines': [
            [4713.17, r"He I: 4713.17", 4708.5, 4709, 4717.35, 4717.85],
            [4009.26, r"He I: 4009.26", 4006, 4006.5, 4012, 4012.5],
            [5875.66, r"He I: 5875.66", 5867, 5867.5, 5882.5, 5883],
            [4026.21, r"He I: 4026.21", 4018.5, 4019, 4031, 4031.5]
        ],
        'Doppler_guess': -2.8
    }

    _4U1700_37 = {
        'lines': [
            [4713.17, r"He I: 4713.17", 4708.5, 4709, 4717.35, 4717.85],
            [4009.26, r"He I: 4009.26", 4006, 4006.5, 4012, 4012.5],
            # [5875.66, r"He I: 5875.66", 5867, 5867.5, 5882.5, 5883],
            [4026.21, r"He I: 4026.21", 4018.5, 4019, 4031, 4031.5]
        ],
        'Doppler_guess': 0.6
    }

    _LMCX_4 = {
        'lines': [
            [4713.17, r"He I: 4713.17", 4708.5, 4709, 4717.35, 4717.85],
            [4009.26, r"He I: 4009.26", 4006, 4006.5, 4012, 4012.5],
            [5875.66, r"He I: 5875.66", 5867, 5867.5, 5882.5, 5883],
            [4026.21, r"He I: 4026.21", 4018.5, 4019, 4031, 4031.5]
        ],
        'Doppler_guess': -5
    }

    _VelaX_1 = {
        'lines': [
            [4713.17, r"He I: 4713.17", 4708.5, 4709, 4717.35, 4717.85],
            [4009.26, r"He I: 4009.26", 4006, 4006.5, 4012, 4012.5],
            # [5875.66, r"He I: 5875.66", 5867, 5867.5, 5882.5, 5883],
            [4026.21, r"He I: 4026.21", 4018.5, 4019, 4031, 4031.5]
        ],
        'Doppler_guess': 0.3
    }

    line_dict = {'4U1538-52': _4U1538_52, 'Cen X-3': _CenX_3, 'SMC X-1': _SMCX_1,
                 '4U1700-37': _4U1700_37, 'LMC X-4': _LMCX_4, 'Vela X-1': _VelaX_1}
    
    for key, lines in line_dict.items():
        for line in lines['lines']:
            for i in range(2,6):
                line[i] += vrad / 3E5 * line[0]

    return line_dict[object_name]



def lines_model_fit(object_name:str)->dict:
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
            [4713.17, r"He I: 4713.17", 4705, 4708, 4713.75, 4715],
            [4009.26, r"He I: 4009.26", 4003, 4005, 4009, 4011],
            [5875.66, r"He I: 5875.66", 5865, 5868, 5880, 5883],
            [4026.21, r"He I: 4026.21", 4018.5, 4022, 4026.4, 4031.5],

            [4921.93, r"He I: 4921.93", 4910, 4916, 4922.4, 4932],
            [4387.93, r"He I: 4387.93", 4381, 4383.75, 4388.20, 4393],
            [4340.46, r"H$\gamma$: 4340.46", 4325, 4335, 4340.7, 4347],
            [4686, r"He II: 4686", 4678, 4681, 4685.75, 4689],
            [5411.53, r"He II: 5411.53", 5405, 5406, 5412.25, 5413.5]
        ],
        'Doppler_guess': 2.8
    }

    _CenX_3 = {
        'lines': [
            [4713.17, r"He I: 4713.17", 4707, 4709, 4718, 4720],
            [4009.26, r"He I: 4009.26", 4006, 4007, 4012, 4012.5],
            [5875.66, r"He I: 5875.66", 5867, 5870, 5880, 5881],
            [4026.21, r"He I: 4026.21", 4020, 4022, 4030, 4031.5],

            [5411.53, r"He II: 5411.53", 5405.75, 5407, 5417.5, 5423],
            [4340.46, r"H$\gamma$: 4340.46", 4326, 4335.6, 4345.4, 4355],
            [4686, r"He II: 4686", 4681.4, 4682.2, 4689, 4691.4],
            [4921.93, r"He I: 4921.93", 4907, 4917.8, 4926.2, 4934]
        ],
        'Doppler_guess': 0
    }

    _SMCX_1 = {
        'lines': [
            [4713.17, r"He I: 4713.17", 4710, 4713, 4719.5, 4721],
            [4009.26, r"He I: 4009.26", 4007, 4009, 4014.5, 4016],
            # [5875.66, r"He I: 5875.66", 5874, 5875.5, 5883.25, 5885],
            # [4026.21, r"He I: 4026.21", 4022.5, 4025, 4032.5, 4034]

            [5411.53, r"He II: 5411.53", 5400, 5412, 5418.5, 5424],
            [4713.17, r"He I: 4713.17", 4710, 4713.3, 4719.7, 4724],
            [4340.46, r"H$\gamma$: 4340.46", 4335, 4339.5, 4347.5, 4350],
            [4686, r"He II: 4686", 4686.1, 4686.4, 4691.3, 4691.9],
            [5015.68, r"He I: 5015.68", 5010, 5015.5, 5022.5, 5029]
        ],
        'Doppler_guess': -2.8
    }

    _4U1700_37 = {
        'lines': [
            # [4713.17, r"He I: 4713.17", 4708.5, 4709, 4715.5, 4717],
            # [4009.26, r"He I: 4009.26", 4005, 4006, 4011, 4012],
            # [5875.66, r"He I: 5875.66", 5867, 5867.5, 5882.5, 5883],
            [4026.21, r"He I: 4026.21", 4017.5, 4020, 4028, 4030],

            [4340.46, r"H$\gamma$: 4340.46", 4333.9, 4334.4, 4343.8, 4344.5],
            # [4686, r"He II: 4686", ],
            [4541, r"He II 4541", 4535.4, 4536.4, 4544.25, 4545]

        ],
        'Doppler_guess': 0.6
    }

    _LMCX_4 = {
        'lines': [
            # [4713.17, r"He I: 4713.17", 4708.5, 4709, 4717.35, 4717.85],
            [4009.26, r"He I: 4009.26", 4010, 4011, 4016, 4017],
            # [5875.66, r"He I: 5875.66", 5874, 5877.5, 5888, 5889.5],
            [4026.21, r"He I: 4026.21", 4022.5, 4025, 4034.5, 4036],

            [5411.53, r"He II: 5411.53", 5405, 5412.5, 5422.3, 5427],
            [4340.46, r"H$\gamma$: 4340.46", 4335, 4339.6, 4350, 4354],
            [4686, r"He II: 4686", 4686.5, 4687.25, 4695, 4695.2],
            [4921.93, r"He I: 4921.93", 4910, 4922.1, 4932.2, 4946]
        ],
        'Doppler_guess': -5
    }

    _VelaX_1 = {
        'lines': [
            [4713.17, r"He I: 4713.17", 4708.5, 4711, 4715, 4717],
            [4009.26, r"He I: 4009.26", 4006, 4006.5, 4012, 4013],
            # [5875.66, r"He I: 5875.66", 5867, 5867.5, 5882.5, 5883],
            [4026.21, r"He I: 4026.21", 4018.5, 4020, 4030, 4031.5],

            [4921.93, r"He I: 4921.93", 4916, 4918.5, 4926, 4932],
            [5411.53, r"He II: 5411.53", 5406.8, 5407.5, 5413.75, 5416.5]
        ],
        'Doppler_guess': 0.3
    }

    line_dict = {'4U1538-52': _4U1538_52, 'Cen X-3': _CenX_3, 'SMC X-1': _SMCX_1,
                 '4U1700-37': _4U1700_37, 'LMC X-4': _LMCX_4, 'Vela X-1': _VelaX_1}

    return line_dict[object_name]



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



"""
DOPPLER SHIFT
"""
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



def determine_radial_velocity(spectra: list, lines: dict, guassian: callable, object_name:str, save=False)->list:
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
    if save:
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
    print(f"The radial velocities are:\n{doppler_shift}")
    return np.mean(doppler_shift)



"""
MODEL FITTING
"""
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
    chi2 = {}
    chi2_perline = {}

    for key, model in models.items():
        # Chi-squared parameter
        chi2[key] = 0
        # Chi-squared per line
        chi2_perline[key] = {}

        for line in lines['lines']:
            # Rest wavelength of the spectral line
            central_wavelength = line[0]

            # Select the spectrum that contains the spectral line
            wav, flux = select_spectrum(spectra, central_wavelength)
            SNR_value = select_SNR(wav, SNR)

            # Extract the spectral line from the spectrum
            wav, flux = extract_spectrum_within_range(wav, flux, line[2], line[5])
            wav_cont, flux_cont = extract_continuum(wav, flux, line[2], line[5], line[3], line[4])
            wav_line, flux_line = extract_spectrum_within_range(wav, flux, line[3], line[4])
            # Extract the line from the model
            wav_model = model['WAVELENGTH']
            flux_model = model['FLUX']
            # Dopplershift the model
            wav_model = doppler_shift_spectrum(wav_model, vrad)
            wav_model, flux_model = extract_spectrum_within_range(np.array(wav_model), np.array(flux_model), 
                                                                    line[2], line[5])


            # Linear fit to continuum
            cont_fit = np.poly1d(np.polyfit(wav_cont, flux_cont, 1))
            # Normalize spectrum
            flux /= cont_fit(wav)
            flux_cont /= cont_fit(wav_cont)
            flux_line /= cont_fit(wav_line)


            # Apply doppler broadening
            wav_model, flux_model = pyasl.equidistantInterpolation(wav_model, flux_model, "2x")
            flux_model = pyasl.rotBroad(wav_model, flux_model, 0.0, vsini)


            # Calculate chi-squared for this line
            chi2_value = chi_squared(wav_model, flux_model, wav_line, flux_line, SNR_value)
            # Keep track of the total chi-squared for all lines
            chi2[key] += chi2_value
            # Save for each line the chi-squared individually
            chi2_perline[key][line[1]] = chi2_value

        # Devide the total chi-squared by the number of lines.
        chi2[key] /= len(lines['lines'])

    print("DONE", end='', flush=True)
    return chi2, chi2_perline



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



"""
PLOTTING RESULTS
"""
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
    lines = lines['lines']
    model = models[best_model]

    num_plots = len(lines)
    num_rows = (num_plots - 1) // 4 + 1  # Calculate the number of rows needed

    fig, axes = plt.subplots(num_rows, 4, figsize=(15, num_rows * 4))

    # Flatten axes if necessary
    if num_rows == 1:
        axes = [axes]

    for i, ax_row in enumerate(axes):

        for j, ax in enumerate(ax_row):
            plot_index = i * 4 + j

            if plot_index < num_plots:

                # Rest wavelength of the spectral line
                central_wavelength = lines[plot_index][0]

                # Select the spectrum that contains the spectral line
                wav, flux = select_spectrum(spectra, central_wavelength)

                # Extract the spectral line from the spectrum
                wav, flux = extract_spectrum_within_range(wav, flux, lines[plot_index][2], lines[plot_index][5])
                wav_cont, flux_cont = extract_continuum(wav, flux, lines[plot_index][2], lines[plot_index][5], 
                                                        lines[plot_index][3], lines[plot_index][4])
                wav_line, flux_line = extract_spectrum_within_range(wav, flux, lines[plot_index][3], lines[plot_index][4])
                # Extract the line from the model
                wav_model = model['WAVELENGTH']
                flux_model = model['FLUX']
                # Dopplershift the model
                wav_model = doppler_shift_spectrum(wav_model, vrad)
                wav_model, flux_model = extract_spectrum_within_range(np.array(wav_model), np.array(flux_model), 
                                                                      lines[plot_index][2], lines[plot_index][5])

                # Linear fit to continuum
                cont_fit = np.poly1d(np.polyfit(wav_cont, flux_cont, 1))
                # Normalize spectrum
                flux /= cont_fit(wav)
                flux_cont /= cont_fit(wav_cont)
                flux_line /= cont_fit(wav_line)


                # Apply doppler broadening
                wav_model, flux_model = pyasl.equidistantInterpolation(wav_model, flux_model, "2x")
                flux_model = pyasl.rotBroad(wav_model, flux_model, 0.0, vsini)


                ax.plot(wav, flux, color='blue', alpha=0.5)
                ax.plot(wav_line, flux_line, color='orange', alpha=0.5)
                ax.plot(wav_model, flux_model, color='green')

                # Annotate each line with text vertically
                ax.set_xlabel(r"Wavelength ($\AA$)", fontsize=12)
                ax.set_ylabel(r"Normalised flux", fontsize=12)
                ax.set_title(lines[plot_index][1], fontsize=12)
                ax.grid(alpha=0.25)

    plt.suptitle(f'Best model: {best_model}', fontsize=15)
    plt.tight_layout()

    if save:
        plt.savefig(save)

    plt.show()

    return



def plot_models_over_lines(spectra: list, models:dict, lines:dict, vrad:float, vsini:float, best_model:str, save=False)->None:
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
    lines = lines['lines']

    num_plots = len(lines)
    num_rows = (num_plots - 1) // 3 + 1  # Calculate the number of rows needed

    fig, axes = plt.subplots(num_rows, 3, figsize=(15, num_rows * 5))

    # Flatten axes if necessary
    if num_rows == 1:
        axes = [axes]

    for i, ax_row in enumerate(axes):

        for j, ax in enumerate(ax_row):
            plot_index = i * 3 + j

            if plot_index < num_plots:

                # Rest wavelength of the spectral line
                central_wavelength = lines[plot_index][0]

                # Select the spectrum that contains the spectral line
                wav, flux = select_spectrum(spectra, central_wavelength)

                # Extract the spectral line from the spectrum
                wav, flux = extract_spectrum_within_range(wav, flux, lines[plot_index][2], lines[plot_index][5])
                wav_cont, flux_cont = extract_continuum(wav, flux, lines[plot_index][2], lines[plot_index][5], 
                                                        lines[plot_index][3], lines[plot_index][4])
                wav_line, flux_line = extract_spectrum_within_range(wav, flux, lines[plot_index][3], lines[plot_index][4])


                # Linear fit to continuum
                cont_fit = np.poly1d(np.polyfit(wav_cont, flux_cont, 1))
                # Normalize spectrum
                flux /= cont_fit(wav)
                flux_cont /= cont_fit(wav_cont)
                flux_line /= cont_fit(wav_line)


                # Plot all models
                for key, model in models.items():
                    # Extract the line from the model
                    wav_model = model['WAVELENGTH']
                    flux_model = model['FLUX']
                    # Dopplershift the model
                    wav_model = doppler_shift_spectrum(wav_model, vrad)
                    wav_model, flux_model = extract_spectrum_within_range(np.array(wav_model), np.array(flux_model), 
                                                                        lines[plot_index][2], lines[plot_index][5])

                    # Apply doppler broadening
                    wav_model, flux_model = pyasl.equidistantInterpolation(wav_model, flux_model, "2x")
                    flux_model = pyasl.rotBroad(wav_model, flux_model, 0.0, vsini)

                    if key == best_model and max(flux_model) < 5:
                        # Plot model
                        ax.plot(wav_model, flux_model, label=f'Best: {key}', color='black', linestyle='--')
                    elif max(flux_model) < 5:
                        ax.plot(wav_model, flux_model, label=f'{key}')


                ax.plot(wav, flux, color='blue', alpha=0.5)
                ax.plot(wav_line, flux_line, color='orange', alpha=0.5)

                # Annotate each line with text vertically
                ax.set_xlabel(r"Wavelength ($\AA$)", fontsize=12)
                ax.set_ylabel("Normalised flux", fontsize=12)
                ax.set_title(lines[plot_index][1], fontsize=12)
                ax.grid(alpha=0.25)

                if len(models) < 10:
                    ax.legend(fontsize=8)

    plt.suptitle('All models', fontsize=15)
    plt.tight_layout()

    if save:
        plt.savefig(save)

    plt.show()

    return