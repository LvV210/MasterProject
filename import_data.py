import pandas as pd


def supergiant_stellar_parameters():
    return pd.read_excel('SupergiantsSpectralParameters.xlsx', header=0, na_values=None)


def HMXB_parameters():
    return pd.read_excel('HMXBparameters.xlsx', header=0, na_values=None)


def HMXB_parameters_Kaper():
    return pd.read_excel('HMXBkaper.xlsx', header=0, na_values=None)


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
