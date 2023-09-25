"""
A table of all parameters of High Mass X-ray Binaries.
{"id": None, "spectraltype": None, "ruwe": None, "periodorbit": None, "spinperiod": None, "eclipseduration": None, "RV": None, "Mopt": None, "Ropt": None, "Mx": None, "parallax": None, "distance": None, "distanceBJ": None, "luminosity": None}
"""
import pandas as pd
import numpy as np
import webbrowser



def HMXB_parameters():
    """

    """
    object_2s0114 = {"id": "2S0114+650", "spectraltype": None, "ruwe": None, "periodorbit": None, "spinperiod": None, "eclipseduration": None, "RV": None, "Mopt": None, "Ropt": None, "Mx": None, "parallax": None, "distance": None, "distanceBJ": None, "luminosity": None}
    SMCX1 = {"id": "SMC X-1", "spectraltype": None, "ruwe": None, "periodorbit": None, "spinperiod": None, "eclipseduration": None, "RV": None, "Mopt": None, "Ropt": None, "Mx": None, "parallax": None, "distance": None, "distanceBJ": None, "luminosity": None}
    LMCX4 = {"id": "LMC X-4", "spectraltype": None, "ruwe": None, "periodorbit": None, "spinperiod": None, "eclipseduration": None, "RV": None, "Mopt": None, "Ropt": None, "Mx": None, "parallax": None, "distance": None, "distanceBJ": None, "luminosity": None}
    VelaX1 = {"id": "Vela X-1", "spectraltype": None, "ruwe": None, "periodorbit": None, "spinperiod": None, "eclipseduration": None, "RV": None, "Mopt": None, "Ropt": None, "Mx": None, "parallax": None, "distance": None, "distanceBJ": None, "luminosity": None}
    CenX3 = {"id": "Cen X-3", "spectraltype": None, "ruwe": None, "periodorbit": None, "spinperiod": None, "eclipseduration": None, "RV": None, "Mopt": None, "Ropt": None, "Mx": None, "parallax": None, "distance": None, "distanceBJ": None, "luminosity": None}
    GX3012 = {"id": "GX301-2", "spectraltype": None, "ruwe": None, "periodorbit": None, "spinperiod": None, "eclipseduration": None, "RV": None, "Mopt": None, "Ropt": None, "Mx": None, "parallax": None, "distance": None, "distanceBJ": None, "luminosity": None}
    object_4U153852 = {"id": "4U1538-52", "spectraltype": None, "ruwe": None, "periodorbit": None, "spinperiod": None, "eclipseduration": None, "RV": None, "Mopt": None, "Ropt": None, "Mx": None, "parallax": None, "distance": None, "distanceBJ": None, "luminosity": None}
    object_4U170037 = {"id": "4U1700-37", "spectraltype": None, "ruwe": None, "periodorbit": None, "spinperiod": None, "eclipseduration": None, "RV": None, "Mopt": None, "Ropt": None, "Mx": None, "parallax": None, "distance": None, "distanceBJ": None, "luminosity": None}
    object_4U190709 = {"id": "4U1907+09", "spectraltype": None, "ruwe": None, "periodorbit": None, "spinperiod": None, "eclipseduration": None, "RV": None, "Mopt": None, "Ropt": None, "Mx": None, "parallax": None, "distance": None, "distanceBJ": None, "luminosity": None}
    LMCX1 = {"id": "LMC X-1", "spectraltype": None, "ruwe": None, "periodorbit": None, "spinperiod": None, "eclipseduration": None, "RV": None, "Mopt": None, "Ropt": None, "Mx": None, "parallax": None, "distance": None, "distanceBJ": None, "luminosity": None}
    CygX1 = {"id": "Cyg X-1", "spectraltype": None, "ruwe": None, "periodorbit": None, "spinperiod": None, "eclipseduration": None, "RV": None, "Mopt": None, "Ropt": None, "Mx": None, "parallax": None, "distance": None, "distanceBJ": None, "luminosity": None}



if __name__ == "__main__":
    HMXB_parameters()
    
    file_path = 'HMXBparameters.xlsx'  # Replace with the path to your XLSX file

    # Read the XLSX file into a DataFrame, setting empty cells to None
    df = pd.read_excel(file_path, header=0, na_values=None)

    df.style

    # html = df.to_html()

    # # Save the html to a temporary file
    # with open('temp.html', 'w') as f:
    #     f.write(html)

    # # Open the file in a new tab
    # webbrowser.open('temp.html', new=2)
