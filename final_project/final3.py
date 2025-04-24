from spectralData import SpectralData
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np 
from scipy.fftpack import *
import pandas as pd 
import os 
class SpectralProcessor:
    def __init__(self):
        print("\nWhat type of data are you using?")
        print("1. IR - Transmission")
        print("2. IR - Absorbance")
        print("3. Raman")
        print("4. UV-Vis")

        self.choice = input("Select an option (1-4): ")

        # Assign correct labels based on choice
        self.labels = {
            '1': ("Wavenumber (cm$^{-1}$)", "Transmittance (%)", True),
            '2': ("Wavenumber (cm$^{-1}$)", "Absorbance", True),
            '3': ("Raman Shift (cm$^{-1}$)", "Intensity (counts)", False),
            '4': ("Wavelength (nm)", "Absorbance", False),
        }

        self.xlabel, self.ylabel, self.invert_x = self.labels.get(self.choice, ("X-axis", "Y-axis", False))

        self.file_path = input("Enter the path to your spectral data file: ")
        try:
            self.data = SpectralData(self.file_path)  # Load spectral data
        except Exception as e:
            print(f"Error loading file: {e}")
            return

        self.processing_steps = []  # Stores preprocessing steps
    
    def fourier_transform(self):
        spectral_data = pd.read_csv(self.file_path)  # Assuming spc contains the spectral data
        n = len(spectral_data)
        fhat = fft(spectral_data, n)  # Compute the FFT
        fhat = np.abs(fhat)  # Get the magnitude
        plt.plot(fhat)  # Plot the FFT
        plt.title("FFT of the dataset")
        plt.xlabel("Frequency")
        plt.ylabel("Magnitude")
        plt.show()
 
    def plot_2d_data(self):
            """Plots spectral data in 2D."""
            plot_data = np.abs(self.data.spc) if np.iscomplexobj(self.data.spc) else self.data.spc  # Remove .values

            plt.figure(figsize=(10, 6))
            plt.plot(plot_data.transpose())  # Assuming the data is 2D
            plt.title("Processed Spectral Data")
            plt.xlabel(self.xlabel)
            plt.ylabel(self.ylabel)

            if self.invert_x:
                plt.gca().invert_xaxis()

            plt.grid(True)
            plt.show()


    def plot_3d_data(self):
        """Plots spectral data in 3D."""
        df = self.data.spc.transpose()
        x = df.columns  
        y = df.index  
        z = df.values  

        fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale="Viridis")])

        fig.update_layout(
            title="Interactive 3D Spectral Plot",
            scene=dict(
                xaxis_title=self.xlabel,
                yaxis_title="Spectrum Index",
                zaxis_title=self.ylabel,
            )
        )

        fig.show()
    

    def process_spectra(self):
        """Step 3: Ask the user to select preprocessing steps."""
        while True:
            print("\nSelect preprocessing steps:")
            print("1. Trim")
            print("2. Baseline Correction")
            print("3. Smoothing")
            print("4. Normalization")
            print("5. Centering")
            print("6. Derivative")
            print("7. Dataset Operations")
            print("8. Apply Selected Steps & Plot")
            print("9. Exit")
            print("0. Fourier Transform")

            choice = input("Choose an option (1-9): ")

            if choice == '1':  
                print("1. Trim")
                print("2. Inverse Trim")
                sub_choice = input("Choose (1 or 2): ")
                start = float(input("Enter start value: "))
                end = float(input("Enter end value: "))
                self.processing_steps.append(("trim" if sub_choice == '1' else "invtrim", (start, end)))

            elif choice == '2':  
                print("1. AsLS")
                print("2. Polyfit")
                print("3. Pearson")
                sub_choice = input("Choose (1-3): ")
                if sub_choice == '1':
                    penalty = float(input("Penalty: "))
                    asymmetry = float(input("Asymmetry: "))
                    iterations = int(input("Iterations: "))
                    self.processing_steps.append(("AsLS", (penalty, asymmetry, iterations)))
                elif sub_choice == '2':
                    order = int(input("Polynomial order: "))
                    iterations = int(input("Iterations: "))
                    self.processing_steps.append(("polyfit", (order, iterations)))
                elif sub_choice == '3':
                    u = int(input("U value: "))
                    v = int(input("V value: "))
                    self.processing_steps.append(("pearson", (u, v)))

            elif choice == '3':  
                print("1. Rolling")
                print("2. Savitzky-Golay")
                sub_choice = input("Choose (1 or 2): ")
                if sub_choice == '1':
                    window = int(input("Window size: "))
                    self.processing_steps.append(("rolling", (window,)))
                elif sub_choice == '2':
                    window = int(input("Window size: "))
                    poly_order = int(input("Polynomial order: "))
                    self.processing_steps.append(("SGSmooth", (window, poly_order)))
            
            elif choice == '4':  
                print("1. SNV")
                print("2. MSC")
                print("3. Detrend")
                sub_choice = input("Choose (1-3): ")
                if sub_choice == '1':
                    self.processing_steps.append(("snv", ()))
                elif sub_choice == '2':
                    ref = int(input("Reference spectrum index (0-based): "))
                    self.processing_steps.append(("msc", (ref,)))
                elif sub_choice == '3':
                    order = int(input("Polynomial order: "))
                    self.processing_steps.append(("detrend", (order,)))

            elif choice == '5':  
                print("1. Mean Centering")
                print("2. Last Point Centering")
                sub_choice = input("Choose (1 or 2): ")
                if sub_choice == '1':
                    self_val = int(input("Enter self value: "))
                    self.processing_steps.append(("mean_center", (self_val,)))
                elif sub_choice == '2':
                    self.processing_steps.append(("lastpoint", ()))

            elif choice == '6':  
                print("1. SG Derivative")
                print("3. Wavelet")
                sub_choice = input("Choose (1-3): ")
                if sub_choice == '1':
                    window = int(input("Window size: "))
                    poly_order = int(input("Polynomial order: "))
                    deriv_order = int(input("Derivative order: "))
                    self.processing_steps.append(("SGDeriv", (window, poly_order, deriv_order)))

                # elif sub_choice == '3':  # Wavelet Transform
                #     wavelet = input("Enter wavelet type (default: 'db1'): ") or "db1"
                #     level = input("Enter decomposition level (leave blank for max level): ")
                #     level = int(level) if level.strip() else None
                #     self.processing_steps.append(("wavelet_transform", (wavelet, level)))
            
            elif choice == '7':
                print("1. Subtract")
                print("2. Reset")
                sub_choice = input("Choose (1 or 2): ")
                if sub_choice == '1':
                    spectra_index = int(input("Spectra index: "))
                    self.processing_steps.append(("subtract", (spectra_index,)))
                elif sub_choice == '2':
                    self.processing_steps.append(("reset", ()))
            
            elif choice == '8':  
                print("\nApplying selected preprocessing steps...\n")
                for method, params in self.processing_steps:
                    if hasattr(self.data, method):
                        getattr(self.data, method)(*params)
                        print(f"Applied {method} with parameters {params}.")
                    else:
                        print(f"Error: {method} is not a valid function in SpectralData.")

                while True:
                    print("\nChoose a plotting method:")
                    print("1. 2D Plot")
                    print("2. 3D Plot")
                    plot_choice = input("Enter choice (1 or 2): ")
                    if plot_choice == '1':
                        self.plot_2d_data()
                        break
                    elif plot_choice == '2':
                        self.plot_3d_data()
                        break
                    else:
                        print("Invalid choice. Please select 1 or 2.")

                self.processing_steps.clear()  

            elif choice == '9':  
                print("Exiting...")
                break
                
            else:
                print("Invalid choice. Please select a valid option.")

    def run(self):
        """Runs the entire workflow."""
        self.process_spectra()


if __name__ == "__main__":
    processor = SpectralProcessor()
    processor.run()
