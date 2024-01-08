# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from astropy.io import fits

# Load the FITS file
file_path1 = "UVES/CenX_3/ADP.2021-08-26T17_24_42.172.fits"
file_path2 = "UVES/CenX_3/ADP.2021-08-26T17_24_42.195.fits"
data1 = fits.getdata(file_path1)
data2 = fits.getdata(file_path2)

# Extract relevant data
wavelength1 = data1['WAVE']
flux1 = data1['FLUX']
err1 = data1['FLUX_REDUCED']

wavelength2 = data2['WAVE']
flux2 = data2['FLUX']
err2 = data2['FLUX_REDUCED']

print(len(wavelength1[0]))
print(len(flux1[0]))
print(data1.columns.names)

# List of colors
colors = [
    'blue',
    'orange',
    'green',
    'red',
    'purple',
    'brown',
    'pink',
    'gray',
    'olive',
    'cyan',
    'lime',
    'teal',
    'indigo',
    'maroon',
    'navy',
    'gold',
    'darkorange',
    'darkgreen',
    'darkred',
    'darkblue'
]

# Plot the spectrum
plt.figure(figsize=(10, 5))
plt.plot(wavelength1[0], flux1[0], label='UVES Spectrum')
plt.plot(wavelength2[0], flux2[0], label='UVES Spectrum')
plt.xlabel('Wavelength (Angstroms)')
plt.ylabel('Flux')
plt.title('UVES Spectrum')
plt.ylim(0, 250)

# Plot wavelengths of spectral lines
plt.vlines(4026, -1000, 1000, label="HeI+II 4026", color=colors[0])
plt.vlines(4200, -1000, 1000, label="HeII 4200", color=colors[1])
plt.vlines(4686, -1000, 1000, label="HeII 4686", color=colors[2])
plt.vlines(4634, -1000, 1000, label="NIII 4634-40-42 (emission)", color=colors[3])
plt.vlines(4144, -1000, 1000, label="HeI 4144", color=colors[4])
plt.vlines(4388, -1000, 1000, label="HeI 4388", color=colors[5])
plt.vlines(4542, -1000, 1000, label="HeII 4542", color=colors[6])
plt.vlines(4552, -1000, 1000, label="SiII 4552", color=colors[7])
plt.vlines(4686, -1000, 1000, label="HeII 4686", color=colors[8])
plt.vlines(4861, -1000, 1000, label="Hb 4861", color=colors[9])
plt.vlines(5016, -1000, 1000, label="HeI 5016", color=colors[10])
plt.vlines(5876, -1000, 1000, label="HeI 5876", color=colors[11])
plt.vlines(5890, -1000, 1000, label="NaI 5890", color=colors[12])
plt.vlines(5896, -1000, 1000, label="NaI 5896", color=colors[13])
plt.vlines(6527, -1000, 1000, label="HeII 6527", color=colors[14])
plt.vlines(6563, -1000, 1000, label="Ha 6563", color=colors[15])

plt.legend()

plt.show()