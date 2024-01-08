# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 15:02:34 2024

@author: luukv
"""

import os
from astropy.io import fits
import matplotlib.pyplot as plt

folder_path = "C:/Users/luukv/Documenten/NatuurSterrkenkundeMasterProject/CodeMP/MasterProject/UVES/4U1538_52/archive/"  # Replace with the actual path to your folder

# Get a list of all files in the folder
all_files = os.listdir(folder_path)

# Filter files that start with "ADP"
adp_files = [file for file in all_files if file.startswith("ADP")]

# Print the list of ADP files
print("List of files starting with ADP:")
for file in adp_files:
    print(file)

# Stack the spectra
# Load the FITS file
flux = []
for file in adp_files:
    file_path = folder_path + file
    data = fits.getdata(file_path)
    wavelength = data['WAVE'][0]
    print(wavelength[0], wavelength[-1])
    flux.append(data['FLUX'][0])
    # plt.plot(wavelength, data['FLUX'][0])

# Use zip to pair corresponding elements of the inner lists
flux_stacked = [sum(x) / len(adp_files) for x in zip(*flux)]

plt.plot(flux_stacked, color='green')
plt.plot(data['FLUX'][0], color='red')
plt.show()