import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

class SpectrumSelector:
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        self.continuum_intervals = []
        self.line_intervals = []
        self.temp_points = []
        self.selecting_continuum = True
        self.fig, self.ax = plt.subplots()
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.onkey)
        self.ax.plot(self.x_data, self.y_data, label='Spectrum')
        self.ax.set_title('Click to select intervals, press "c" for continuum, "1" for line\nActive: CONTINUUM')
        self.ax.legend()
    
    def onkey(self, event):
        if event.key == 'c':
            self.selecting_continuum = True
            self.ax.set_title('Click to select intervals, press "c" for continuum, "1" for line\nActive: CONTINUUM')

            print("Switched to selecting continuum intervals.")
        elif event.key == '1':
            self.selecting_continuum = False
            self.ax.set_title('Click to select intervals, press "c" for continuum, "1" for line\nActive: LINE')
            print("Switched to selecting line intervals.")
    
    def onclick(self, event):
        if event.inaxes:
            x = event.xdata
            index = (np.abs(self.x_data - x)).argmin()
            self.temp_points.append((index, self.x_data[index]))

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
                print(f"Selected {label} interval: {interval[0][1]} to {interval[1][1]}")
                self.ax.plot(self.x_data[interval[0][0]:interval[1][0]], self.y_data[interval[0][0]:interval[1][0]], color=color, linestyle='--', label=label)
                
                self.temp_points = []
                self.fig.canvas.draw()

    def select_intervals(self):
        plt.show()

    def get_selected_intervals(self):
        return {
            'continuum_intervals': self.continuum_intervals,
            'line_intervals': self.line_intervals
        }

# Example usage:
if __name__ == "__main__":
    x_data = np.linspace(0, 10, 1000)
    y_data = np.sin(x_data) + np.random.normal(0, 0.1, x_data.size)

    selector = SpectrumSelector(x_data, y_data)
    selector.select_intervals()

    selected_intervals = selector.get_selected_intervals()
    print("Selected continuum intervals (index, x):", selected_intervals['continuum_intervals'])
    print("Selected line intervals (index, x):", selected_intervals['line_intervals'])

