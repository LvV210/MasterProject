import os
import shutil

# Specify the path of the source file and the destination directory
objects = ['4U1538_52', '4U1700_37', 'CenX_3', 'LMCX_4', 'SMCX_1', 'VelaX_1']
source_folder = '/mnt/c/Users/luukv/Documenten/NatuurSterrkenkundeMasterProject/CodeMP/MasterProject/ModelFitting/FitResults/'
destination_folder = '/mnt/c/Users/luukv/OneDrive/Masterproject/AllNotes/Results/'

# Get all folders in fitresults
result_folders = folders = [entry for entry in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, entry))]

for HMXB in objects:
    for result_folder in result_folders:
        if HMXB in result_folder:
            # print(os.path.join(destination_folder, HMXB), '\n',os.path.join(source_folder, result_folder, 'BestModel.png'))
            # print('\n\n')
            shutil.copy(os.path.join(source_folder, result_folder, 'BestModel.png'), os.path.join(destination_folder, HMXB, 'MostRecentResult'))
            shutil.copy(os.path.join(source_folder, result_folder, 'AllModelsOverData.png'), os.path.join(destination_folder, HMXB, 'MostRecentResult'))
            shutil.copy(os.path.join(source_folder, result_folder, 'AllModelsOverData(logg).png'), os.path.join(destination_folder, HMXB, 'MostRecentResult'))
            shutil.copy(os.path.join(source_folder, result_folder, 'AllModelsOverData(Teff).png'), os.path.join(destination_folder, HMXB, 'MostRecentResult'))
            shutil.copy(os.path.join(source_folder, result_folder, 'RadialVelocity.png'), os.path.join(destination_folder, HMXB, 'MostRecentResult'))
            shutil.copy(os.path.join(source_folder, result_folder, 'ChiSquaredPerParameter.png'), os.path.join(destination_folder, HMXB, 'MostRecentResult'))
            shutil.copy(os.path.join(source_folder, result_folder, 'ResultPerLine.png'), os.path.join(destination_folder, HMXB, 'MostRecentResult'))