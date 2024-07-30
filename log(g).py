import numpy as np
from astropy.constants import M_sun, G, R_sun


M = 16
R = 13
logg = (np.log10(M * M_sun.value * G.value * 100 ** 3 / (R * R_sun.value * 100)**2))

print(f"log(g) = {round(logg, 2)}")