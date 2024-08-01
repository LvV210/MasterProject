from AllModelFunctions import select_spectrum, import_spectra, save_name_to_object_name
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import shutil
import json
from scipy.interpolate import UnivariateSpline



ALL_LINES = {
         'Hg': 4340.47,
         'Hd': 4101.73,
         'He': 3970.08,
         'Hf': 3889.06,
         'Hn': 3835.4,
         'HeI_4471.50': 4471.5,
         'HeI_4026.21': 4026.21,
         'HeI_3819.62': 3819.62,
         'HeI_3705.02': 3705.02,
         'HeI_3634.25': 3634.25,
         'HeI_3587.27': 3587.27,
         'HeI_4921.93': 4921.93,
         'HeI_4387.93': 4387.93,
         'HeI_4143.76': 4143.76,
         'HeI_4009.26': 4009.26,
         'HeI_4713.17': 4713.17,
         'HeI_5015.68': 5015.68,
         'HeI_5047.74': 5047.74,
         'HeII_5411.53': 5411.53,
         'HeII_4199.83': 4199.83,
         'SiIII_5739': 5739,
         'SiIII_4552': 4552,
         'SiIII_4567': 4567,
         'SiIII4574': 4574,
         'SiIV4116': 4116,
         'SiIV4212': 4212,
         'SiIV4089': 4089}


LINES_4U1538 = {
         'Hg': 4340.47,
         'Hd': 4101.73,
         'Hf': 3889.06,
         'Hn': 3835.4,
         'HeI_4471.50': 4471.5,
         'HeI_4921.93': 4921.93,
         'HeI_4387.93': 4387.93,
         'HeI_4713.17': 4713.17,
         'HeI_5015.68': 5015.68,
         'HeI_5047.74': 5047.74,
         'HeII_5411.53': 5411.53,
         'HeII_4199.83': 4199.83,
         'SiIV4116': 4116,
         'SiIV4212': 4212,
         'SiIV4089': 4089
}


LINES_4U1700 = {
         'Hg': 4340.47,
         'Hd': 4101.73,
         'Hf': 3889.06,
         'Hn': 3835.4,
         'HeI_4471.50': 4471.5,
         'HeI_4026.21': 4026.21,
         'HeII_4199.83': 4199.83}


LINES_Cen = {
         'Hg': 4340.47,
         'Hd': 4101.73,
         'He': 3970.08,
         'Hf': 3889.06,
         'Hn': 3835.4,
         'HeI_4471.50': 4471.5,
         'HeI_4026.21': 4026.21,
         'HeI_4921.93': 4921.93,
         'HeI_4387.93': 4387.93,
         'HeI_4143.76': 4143.76,
         'HeI_4009.26': 4009.26,
         'HeI_4713.17': 4713.17,
         'HeI_5015.68': 5015.68,
         'HeII_5411.53': 5411.53,
         'HeII_4199.83': 4199.83,
         'SiIV4089': 4089}


LINES_LMC = {
         'Hg': 4340.47,
         'Hd': 4101.73,
         'He': 3970.08,
         'Hf': 3889.06,
         'Hn': 3835.4,
         'HeI_4471.50': 4471.5,
         'HeI_4026.21': 4026.21,
         'HeI_4921.93': 4921.93,
         'HeI_4387.93': 4387.93,
         'HeI_4143.76': 4143.76,
         'HeI_4009.26': 4009.26,
         'HeI_4713.17': 4713.17,
         'HeI_5015.68': 5015.68,
         'HeI_5047.74': 5047.74,
         'HeII_5411.53': 5411.53,
         'HeII_4199.83': 4199.83}


LINES_SMC = {
         'Hg': 4340.47,
         'Hd': 4101.73,
         'He': 3970.08,
         'Hf': 3889.06,
         'Hn': 3835.4,
         'HeI_4471.50': 4471.5,
         'HeI_4026.21': 4026.21,
         'HeI_4921.93': 4921.93,
         'HeI_4387.93': 4387.93,
         'HeI_4143.76': 4143.76,
         'HeI_4009.26': 4009.26,
         'HeI_4713.17': 4713.17,
         'HeI_5015.68': 5015.68,
         'HeI_5047.74': 5047.74,
         'HeII_5411.53': 5411.53,
         'HeII_4199.83': 4199.83,
         'SiIV4116': 4116,
         'SiIV4212': 4212,
         'SiIV4089': 4089}


LINES_Vela = {
         'Hg': 4340.47,
         'Hf': 3889.06,
         'Hn': 3835.4,
         'HeI_4471.50': 4471.5,
         'HeI_4026.21': 4026.21,
         'HeI_4921.93': 4921.93,
         'HeI_4387.93': 4387.93,
         'HeI_4143.76': 4143.76,
         'HeI_4009.26': 4009.26,
         'HeI_4713.17': 4713.17,
         'HeI_5015.68': 5015.68,
         'HeI_5047.74': 5047.74,
         'HeII_5411.53': 5411.53,
         'HeII_4199.83': 4199.83,
         'SiIII_4552': 4552,
         'SiIII_4567': 4567,
         'SiIII4574': 4574,
         'SiIV4116': 4116,
         'SiIV4212': 4212,
         'SiIV4089': 4089}



def remove_folder_and_contents(folder_path):
    # Check if the folder exists
    if os.path.exists(folder_path):
        # Iterate over all the contents of the folder
        for root, dirs, files in os.walk(folder_path, topdown=False):
            # Remove all files
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)
            # Remove all directories
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                os.rmdir(dir_path)
        # Remove the main folder itself
        os.rmdir(folder_path)
        print(f"Folder '{folder_path}' and all its contents have been removed.")
    else:
        print(f"Folder '{folder_path}' does not exist.")



class SpectrumSelector:
    def __init__(self, wav, flux, line_label, central_wav, folder_path):
        # Spectrum and central wavelength of line
        self.wav = wav
        self.flux = flux
        self.line_label = line_label
        self.central_wav = central_wav
        self.folder_path = folder_path

        # Intervals like: [((inex, wav), (index, wav)), etc.]
        self.continuum_intervals = []
        self.line_intervals = []
        self.temp_points = []
        self.selecting_continuum = True
        self.fig, self.ax = plt.subplots()
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.onkey)
        self.ax.plot(self.wav, self.flux, label='Spectrum', color='black')
        self.ax.vlines(x=self.central_wav, ymin=min(self.flux), ymax=max(self.flux), color='limegreen', linestyle='--', linewidth=1)
        self.ax.set_title(f'{self.line_label}\nClick to select intervals, press "0" for continuum, "1" for line\nActive: CONTINUUM')
        self.ax.legend()




    def onkey(self, event):
        """
        Switch between continuum- and line selection mode

        Args:
            event (?): String of the key inputted
        """
        if event.key == '0':
            self.selecting_continuum = True
            self.ax.set_title(f'{self.line_label}\nClick to select intervals, press "0" for continuum, "1" for line\nActive: CONTINUUM')

            print("\tSwitched to selecting continuum intervals.")
        elif event.key == '1':
            self.selecting_continuum = False
            self.ax.set_title(f'{self.line_label}\nClick to select intervals, press "0" for continuum, "1" for line\nActive: LINE')
            print("\tSwitched to selecting line intervals.")



    def onclick(self, event):
        """
        Wavelength and index of point selected in the plot is saved.
        When 2 points (1 interval) is selected, the interval is plotted.
        Intervals for continuum and lines are separately handled.

        Args:
            event (?): Coordinates clicked point.
        """
        if event.dblclick and event.inaxes:
            # Read coordinate and get inex. Add point to temporary points.
            x = event.xdata
            index = (np.abs(self.wav - x)).argmin()
            self.temp_points.append((index, self.wav[index]))

            # When 2 points are selected, make it a tuple interval.
            if len(self.temp_points) == 2:
                interval = (self.temp_points[0], self.temp_points[1])

                if self.selecting_continuum:
                    self.continuum_intervals.append(interval)
                    color = 'blue'
                    label = 'Continuum'
                else:
                    self.line_intervals.append(interval)
                    color = 'red'
                    label = 'Line'

                # Mark the selected interval on the plot
                print(f"\tSelected {label} interval: {round(interval[0][1], 2)} to {round(interval[1][1], 2)}")
                self.ax.plot(self.wav[interval[0][0]:interval[1][0]], self.flux[interval[0][0]:interval[1][0]], color=color, linestyle='--', label=label)

                # Reset the temporary point
                self.temp_points = []
                self.fig.canvas.draw()



    def select_intervals(self):
        plt.show()



    def plot_intervals(self):

        # Determine plot interval
        indices = [index for pair in self.continuum_intervals for index, _ in pair]
        lowest_index = min(indices) - 100
        highest_index = max(indices) + 100

        plt.figure(figsize=(12,7))
        # Plot spectrum
        plt.plot(self.wav, self.flux, color='black', linewidth=1, label='spectrum')
        # Plot continuum intervals
        for interval in self.continuum_intervals:
            plt.plot(self.wav[interval[0][0]:interval[1][0]], self.flux[interval[0][0]:interval[1][0]], color='blue', linestyle='-', label='continuum')
        for interval in self.line_intervals:
            plt.plot(self.wav[interval[0][0]:interval[1][0]], self.flux[interval[0][0]:interval[1][0]], color='red', linestyle='-', label='line')

        # Set labels
        plt.xlabel(r"$\lambda$ ($\AA$)", fontsize=12)
        plt.ylabel(r"Flux (arbitrary units)", fontsize=12)
        plt.title(f"Intervals for {self.line_label}", fontsize=15)

        plt.xlim(self.wav[lowest_index], self.wav[highest_index])
        plt.ylim(0, 2 * np.mean(self.flux))
        plt.legend()
        plt.grid(alpha=0.25)
        plt.savefig(self.folder_path + '/Plots/Intervals.png')
        plt.show()


    def normalize_interval(self):
        """
        Normalize the spectrum based on the given continuum.
        Also plot the results and save it.
        """
        # Determine plot interval
        indices = [index for pair in self.continuum_intervals for index, _ in pair]
        lowest_index = min(indices) - 100
        highest_index = max(indices) + 100
        wav = self.wav[lowest_index:highest_index]
        flux = self.flux[lowest_index:highest_index]
        
        # Extract continuum from spectrum
        wav_cont = []
        flux_cont = []
        for interval in self.continuum_intervals:
            interval = sorted(interval, key=lambda x: x[0])
            for x, y in zip(self.wav[interval[0][0]:interval[1][0]], self.flux[interval[0][0]:interval[1][0]]):
                wav_cont.append(x)
                flux_cont.append(y)

        # Use a smoothing spline to fit the continuum
        spline = np.polyfit(wav_cont, flux_cont, 2)
        continuum = np.polyval(spline, wav)
        norm_flux = flux / continuum

        # Extract line and normalize
        wav_norm_line = self.wav[self.line_intervals[0][0][0]:self.line_intervals[0][1][0]]
        flux_norm_line = self.flux[self.line_intervals[0][0][0]:self.line_intervals[0][1][0]] / np.polyval(spline, wav_norm_line)

        # Plot the normalization and the fit
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        ax1.plot(wav, flux, color='black', label='Spectrum')
        ax1.plot(wav_cont, flux_cont, color='blue', label='Continuum')
        ax1.plot(wav, continuum, color='limegreen', label='Continuum fit', linewidth=2, linestyle='--')

        ax1.set_xlim(min(wav_cont) - 1, max(wav_cont) + 1)
        ax1.set_ylim(0, ymax=max(flux_cont) + np.mean(flux_cont) / 10)
        ax1.legend()
        ax1.set_title("Spectrum and continuum fit", fontsize=15)
        ax1.set_xlabel(r"$\lambda$ ($\AA$)", fontsize=12)
        ax1.set_ylabel("Flux (arbitrary units)", fontsize=12)
        ax1.grid(alpha=0.25)

        ax2.set_title("Normalized", fontsize=15)
        ax2.set_xlabel(r"$\lambda$ ($\AA$)", fontsize=12)
        ax2.set_ylabel("Norm. Flux", fontsize=12)
        ax2.plot(wav, norm_flux, color='black')
        ax2.plot(wav_norm_line, flux_norm_line, color='red')
        ax2.grid(alpha=0.25)


        plt.suptitle(f"{self.line_label}")
        plt.tight_layout()
        plt.savefig(self.folder_path + '/Plots/NormFit.png')
        plt.show()

        # Save normalized spectra
        self.wav_norm_line = wav_norm_line.tolist()
        self.flux_norm_line = flux_norm_line.tolist()
        self.wav_norm = wav.tolist()
        self.flux_norm = norm_flux.tolist()



    def save(self):
        """
        Save normalized line and surrounding spectrum
        """
        # Save Line
        with open(self.folder_path + "/Line.json", "w") as file: 
            json.dump({"Wavelength": self.wav_norm_line,
                       "Flux": self.flux_norm_line}, file)
            
        # Save spectrum around Line
        with open(self.folder_path + "/Spectrum.json", "w") as file: 
            json.dump({"Wavelength": self.wav_norm,
                       "Flux": self.flux_norm}, file)

        # Save Continuum intervals
        with open(self.folder_path + "/ContinuumIntervals.txt", "w") as file: 
            file.write(f"{self.continuum_intervals}")

        # Save Line intervals
        with open(self.folder_path + "/LineIntervals.txt", "w") as file: 
            file.write(f"{self.line_intervals}")



    def get_selected_intervals(self):
        return {
            'continuum_intervals': self.continuum_intervals,
            'line_intervals': self.line_intervals
        }




if __name__ == "__main__":

    # Initialize
    object_ = 'VelaX_1'
    LINES = LINES_Vela
    not_satisfied = []

    for line_label, line_wav in LINES.items():
        print(f"{line_label}")
        folder_path = f"/mnt/c/Users/luukv/Documenten/NatuurSterrkenkundeMasterProject/CodeMP/MasterProject/ModelFitting/Lines/{object_}/{line_label}"

        if os.path.exists(folder_path) == True and os.path.isdir(folder_path) == True:
            print(f"\tLine Already EXISTS")

        elif os.path.exists(folder_path) == False and os.path.isdir(folder_path) == False:
            # Make folder for LINE
            os.mkdir(folder_path)
            os.mkdir(folder_path + '/Plots')

            # Get spectrum containing the LINE
            wav, flux = select_spectrum(import_spectra(object_), line_wav)


            """
            START SELECTION
            """
            # Initialize selector
            selector = SpectrumSelector(wav, flux, line_label, line_wav, folder_path)

            selector.select_intervals()


            selected_intervals = selector.get_selected_intervals()
            # print("Selected continuum intervals (index, x):", selected_intervals['continuum_intervals'])
            # print("Selected line intervals (index, x):", selected_intervals['line_intervals'])


            selector.plot_intervals()


            # Ask user if the intervals are satisfactory
            AreYouSatisfied = input("Are the intervals correct (y/n)?")

            while AreYouSatisfied != 'y' and AreYouSatisfied != 'Y' and AreYouSatisfied != 'n' and AreYouSatisfied != 'N':
                if AreYouSatisfied == 'y' or AreYouSatisfied == 'Y':
                    print("\tThat is good to hear")
                elif AreYouSatisfied == 'n' or AreYouSatisfied == 'N':
                    sys.exit("Re-run programm to get better intervals")
                else:
                    # For wrong input
                    print("\tWRONG INPUT")
                    AreYouSatisfied =  input("\tAre the intervals correct (y/n)?")


            selector.normalize_interval()


            AreYouSatisfied = input("\tIs te normalization done correctly?")

            while AreYouSatisfied != 'y' and AreYouSatisfied != 'Y' and AreYouSatisfied != 'n' and AreYouSatisfied != 'N':
                if AreYouSatisfied == 'y' or AreYouSatisfied == 'Y':
                    print("\tThat is good to hear")
                elif AreYouSatisfied == 'n' or AreYouSatisfied == 'N':
                    # Not satisfied --> Redo normalization
                    selector = SpectrumSelector(wav, flux, line_label, line_wav, 
                                                "/mnt/c/Users/luukv/Documenten/NatuurSterrkenkundeMasterProject/CodeMP/MasterProject/ModelFitting/Lines/4U1538-52")
                    selector.select_intervals()
                    selector.normalize_interval()
                    AreYouSatisfied = input("\tIs te normalization done correctly (y/n)?")
                else:
                    # For wrong input
                    print("\tWRONG INPUT")
                    AreYouSatisfied =  input("\tAre the intervals correct (y/n)?")


            selector.save()


            # Check if result is satisfactory and if line should be removed
            AreYouSatisfied = input("\tAre you satisfied with the result (y/n)?")

            if AreYouSatisfied == 'n' or AreYouSatisfied == 'N':
                not_satisfied.append(line_label)

                AreYouSatisfied = input("\tShould this line be deleted (y/n)?")
                if AreYouSatisfied == 'y' or AreYouSatisfied == 'Y':
                    # Remove the folder and its contents
                    remove_folder_and_contents(folder_path)

    print(f"Not satisfied with:\t{not_satisfied}")
