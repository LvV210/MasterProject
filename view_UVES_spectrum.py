# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from astropy.io import fits

# Load the FITS file
file_path = "UVES/4U1538-52/ADP.2020-06-19T09_21_44.324.fits"
data = fits.getdata(file_path)

# Extract relevant data
wavelength = data['WAVE']
flux = data['BGFLUX_REDUCED']

print(len(wavelength[0]))
print(len(flux[0]))
print(data.columns.names)

# Plot the spectrum
plt.figure(figsize=(10, 5))
plt.plot(wavelength[0], flux[0], label='UVES Spectrum')
plt.xlabel('Wavelength (Angstroms)')
plt.ylabel('Flux')
plt.title('UVES Spectrum')
plt.legend()
plt.show()