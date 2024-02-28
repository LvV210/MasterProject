import pandas as pd
import numpy as np
from functions import *


def supergiant_stellar_parameters():
    return pd.read_excel('../tables/SupergiantsSpectralParameters.xlsx', header=0, na_values=None)


def HMXB_parameters():
    return pd.read_excel('../tables/HMXBparameters.xlsx', header=0, na_values=None)


def HMXB_parameters_Kaper():
    return pd.read_excel('../tables/HMXBkaper.xlsx', header=0, na_values=None)


def stellar_params():
    return pd.read_excel('../tables/StellarParam.xlsx', header=0, na_values=None)


def photometric_params():
    return pd.read_excel('../tables/PhotometricParam.xlsx', header=0, na_values=None)


def falenga():
    """
    Import orbital parameters of the 6 HMXB from Falenga et al. (2015)

    Params:
        Semic eclipse angle:    degrees
        a:                      Solar radii
        i:                      degrees

    returns:
        df:     DataFrame of the parameters
    """

    LMCx4 = {"id": "LMC X-4",
             "semi_eclipse_angle": 15.8,
             "semi_eclipse_angle _err": 0.8,
             "a": 14.2,
             "a_err": 0.2,
             "i": 59.3,
             "i_err": 0.9,
             "Mopt": 18,
             "Mopt_err": 1,
             "Ropt": 7.4,
             "Ropt_err": 0.4,
             "P_orbit": 1.40839776,
             "P_spin": 13.5,
             "RL/a": 0.59,
             "RL/a_err": 0.01, 
             "Mx": 1.57,
             "Mx_err": 0.11}
    Cenx3 = {"id": "Cen X-3",
             "semi_eclipse_angle": 27.9,
             "semi_eclipse_angle _err": 0.3,
             "a": 20.2,
             "a_err": 0.4,
             "i": 65,
             "i_err": 1,
             "Mopt": 24,
             "Mopt_err": 1,
             "Ropt": 11.4,
             "Ropt_err": 0.7,
             "P_orbit": 2.087113936,
             "P_spin": 4.8,
             "RL/a": 0.63,
             "RL/a_err": 0.01, 
             "Mx": 1.57,
             "Mx_err": 0.16}
    U1700 = {"id": "4U1700-37",
             "semi_eclipse_angle": 32,
             "semi_eclipse_angle _err": 1,
             "a": 35,
             "a_err": 1,
             "i": 62,
             "i_err": 1,
             "Mopt": 46,
             "Mopt_err": 5,
             "Ropt": 22,
             "Ropt_err": 2,
             "P_orbit":  3.411581,
             "P_spin": None,
             "RL/a": 0.694,
             "RL/a_err": 0.006, 
             "Mx": 1.96,
             "Mx_err": 0.19}
    U1538 = {"id": "4U1538-52",
             "semi_eclipse_angle": 21,
             "semi_eclipse_angle _err": 1,
             "a": 22,
             "a_err": 1,
             "i": 67,
             "i_err": 1,
             "Mopt": 16,
             "Mopt_err": 2,
             "Ropt": 13,
             "Ropt_err": 1,
             "P_orbit": 3.728382,
             "P_spin": 526.8,
             "RL/a": 0.53,
             "RL/a_err": 0.03, 
             "Mx": 1.02,
             "Mx_err": 0.17}
    SMCx1 = {"id": "SMC X-1",
             "semi_eclipse_angle": 23,
             "semi_eclipse_angle _err": 2,
             "a": 27.9,
             "a_err": 0.7,
             "i": 62,
             "i_err": 2,
             "Mopt": 18,
             "Mopt_err": 2,
             "Ropt": 15,
             "Ropt_err": 1,
             "P_orbit": 3.89220909,
             "P_spin": 0.71,
             "RL/a": 0.61,
             "RL/a_err": 0.02, 
             "Mx": 1.21,
             "Mx_err": 0.12}
    Velax1 = {"id": "Vela X-1",
              "semi_eclipse_angle": 30.5,
              "semi_eclipse_angle _err": 0.1,
              "a": 59.6,
              "a_err": 0.7,
              "i": 72.8,
              "i_err": 0.4,
              "Mopt": 26,
              "Mopt_err": 1,
              "Ropt": 29,
              "Ropt_err": 1,
              "P_orbit": 8.964368,
              "P_spin": 283.2,
             "RL/a": 0.595,
             "RL/a_err": 0.006, 
             "Mx": 2.12,
             "Mx_err": 0.16}
    SAX =   {"id": "IGR J18027−2016",
              "semi_eclipse_angle": 31,
              "semi_eclipse_angle _err": 2,
              "a": 33,
              "a_err": 1,
              "i": 72,
              "i_err": 2,
              "Mopt": 22,
              "Mopt_err": 2,
              "Ropt": 18,
              "Ropt_err": 1,
              "P_orbit":  4.5696,
              "P_spin": 139.6,
             "RL/a": 0.61,
             "RL/a_err": 0.02, 
             "Mx": 1.57,
             "Mx_err": 0.25}
    XTE =   {"id": "XTE J1855-026",
              "semi_eclipse_angle": 32,
              "semi_eclipse_angle _err": 1,
              "a": 40,
              "a_err": 1,
              "i": 71,
              "i_err": 2,
              "Mopt": 21,
              "Mopt_err": 2,
              "Ropt": 22,
              "Ropt_err": 2,
              "P_orbit":  6.0724,
              "P_spin": 360.7,
             "RL/a": 0.63,
             "RL/a_err": 0.03, 
             "Mx": 1.41,
             "Mx_err": 0.24}
    EXO =   {"id": "EXO 1722-363",
              "semi_eclipse_angle": 26,
              "semi_eclipse_angle _err": 4,
              "a": 52,
              "a_err": 2,
              "i": 68,
              "i_err": 2,
              "Mopt": 18,
              "Mopt_err": 2,
              "Ropt": 26,
              "Ropt_err": 2,
              "P_orbit": 9.7403,
              "P_spin": 413.9}
    OAO =   {"id": "OAO 1657-415",
              "semi_eclipse_angle": 20.9,
              "semi_eclipse_angle _err": 0.4,
              "a": 51.3,
              "a_err": 0.8,
              "i": 67.9,
              "i_err": 0.9,
              "Mopt": 17.5,
              "Mopt_err": 0.8,
              "Ropt": 25,
              "Ropt_err": 2,
              "P_orbit": 10.44749,
              "P_spin": 37.3}

    df = pd.DataFrame([LMCx4, Cenx3, U1538, U1700, SMCx1, Velax1, SAX, XTE, EXO, OAO])

    return df



def falenga_table7():
    df_falenga7 = pd.read_csv('../tables/falenga_table7.csv', delimiter=',')

    for column in df_falenga7:

        if column != 'Source':
            df_falenga7[column + '_err'] = 0
            for i in range(len(df_falenga7[column])):
                value, err = extract_number_and_error(df_falenga7[column].loc[i])
                df_falenga7.loc[i, column] = float(value)
                df_falenga7.loc[i, column + '_err'] = float(err)

    for column in df_falenga7:
        if column != 'Source':
            for i in range(len(df_falenga7[column])):
                    df_falenga7.loc[i, column] = float(df_falenga7.loc[i, column])

    return df_falenga7.rename(columns={'Source': 'id'})


def BailerJones():
    """
    Import distances to object from Bailer-Jones database
    Change distances to LMC and SMC
    """
    # Import
    BJ = pd.read_csv('../tables/BailerJonesDistances.csv', sep=',', header=0, na_values=None)
    ID = pd.read_excel('../tables/id_converter.xlsx')
    # Merge
    BJ['source_id'] = BJ["source_id"].astype(np.int64)
    ID['source_id'] = ID["source_id"].astype(np.int64)
    BJ = pd.merge(ID, BJ, on='source_id')
    BJ['source_id'] = BJ['source_id'].astype(str)

    # Change distance to LMC and SMC
    BJ.loc[BJ['id'] == 'SMC X-1', ['r_med_photogeo', 'r_med_geo']] = 62440
    BJ.loc[BJ['id'] == 'SMC X-1', ['r_lo_photogeo', 'r_lo_geo']] = 58440
    BJ.loc[BJ['id'] == 'SMC X-1', ['r_hi_photogeo', 'r_hi_geo']] = 62440

    BJ.loc[BJ['id'] == 'LMC X-1', ['r_med_photogeo', 'r_med_geo']] = 48590
    BJ.loc[BJ['id'] == 'LMC X-1', ['r_lo_photogeo', 'r_lo_geo']] = 44590
    BJ.loc[BJ['id'] == 'LMC X-1', ['r_hi_photogeo', 'r_hi_geo']] = 52590

    BJ.loc[BJ['id'] == 'LMC X-4', ['r_med_photogeo', 'r_med_geo']] = 48590
    BJ.loc[BJ['id'] == 'LMC X-4', ['r_lo_photogeo', 'r_lo_geo']] = 44590
    BJ.loc[BJ['id'] == 'LMC X-4', ['r_hi_photogeo', 'r_hi_geo']] = 52590

    return BJ