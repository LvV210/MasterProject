import sys
sys.path.append('/mnt/c/Users/luukv/Documenten/NatuurSterrkenkundeMasterProject/CodeMP/MasterProject')
from functions import import_spectra, extract_spectrum_within_range

object_name = 'VelaX_1'
spectra = import_spectra(object_name)

wavelength = spectra[0][0]
flux = spectra[0][1]

wavelength, flux = extract_spectrum_within_range(wavelength, flux, 3710, 9200)

# Specify the file name
file_name = f'{object_name}.txt'

# Open the file in write mode
with open(file_name, 'w') as file:
    # Write data from lists into the file
    for wl, fl in zip(wavelength, flux):
        file.write(f"{wl} {fl}\n")

print(f"Data saved to {file_name}")