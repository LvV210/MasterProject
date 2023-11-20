import pandas as pd
import numpy as np


def supergiant_stellar_parameters():
    return pd.read_excel('tables/SupergiantsSpectralParameters.xlsx', header=0, na_values=None)


def HMXB_parameters():
    return pd.read_excel('tables/HMXBparameters.xlsx', header=0, na_values=None)


def HMXB_parameters_Kaper():
    return pd.read_excel('tables/HMXBkaper.xlsx', header=0, na_values=None)


def stellar_params():
    return pd.read_excel('tables/StellarParam.xlsx', header=0, na_values=None)


def photometric_params():
    return pd.read_excel('tables/PhotometricParam.xlsx', header=0, na_values=None)


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

    LMCx4 = {"id": "LMC X-4","semi_eclipse_angle": 15.8,"semi_eclipse_angle _err": 0.8,"a": 14.2,"a_err": 0.2,"i": 59.3,"i_err": 0.9}
    Cenx3 = {"id": "Cen X-3","semi_eclipse_angle": 27.9,"semi_eclipse_angle _err": 0.3,"a": 20.2,"a_err": 0.4,"i": 65,"i_err": 1}
    U1700 = {"id": "4U1700-37","semi_eclipse_angle": 32,"semi_eclipse_angle _err": 1,"a": 35,"a_err": 1,"i": 62,"i_err": 1}
    U1538 = {"id": "4U1538-52","semi_eclipse_angle": 21,"semi_eclipse_angle _err": 1,"a": 22,"a_err": 1,"i": 67,"i_err": 1}
    SMCx1 = {"id": "SMC X-1","semi_eclipse_angle": 23,"semi_eclipse_angle _err": 2,"a": 27.9,"a_err": 0.7,"i": 62,"i_err": 2}
    Velax1 = {"id": "Vela X-1","semi_eclipse_angle": 30.5,"semi_eclipse_angle _err": 0.1,"a": 59.6,"a_err": 0.7,"i": 72.8,"i_err": 0.4}

    df = pd.DataFrame([LMCx4, Cenx3, U1538, U1700, SMCx1, Velax1])

    return df


def BailerJones():
    """
    Import distances to object from Bailer-Jones database
    Change distances to LMC and SMC
    """
    # Import
    BJ = pd.read_csv('tables/BailerJonesDistances.csv', sep=',', header=0, na_values=None)
    ID = pd.read_excel('tables/id_converter.xlsx')
    # Merge
    BJ['source_id'] = BJ["source_id"].astype(np.int64)
    ID['source_id'] = ID["source_id"].astype(np.int64)
    BJ = pd.merge(ID, BJ, on='source_id')
    BJ['source_id'] = BJ['source_id'].astype(str)

    # Change distance to LMC and SMC
    BJ.loc[BJ['id'] == 'SMC X-1', ['r_med_photogeo', 'r_med_geo']] = 64000
    BJ.loc[BJ['id'] == 'LMC X-1', ['r_med_photogeo', 'r_med_geo']] = 50000
    BJ.loc[BJ['id'] == 'LMC X-4', ['r_med_photogeo', 'r_med_geo']] = 50000

    return BJ