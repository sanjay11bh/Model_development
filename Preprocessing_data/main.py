from spectral import SpectralData
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def plot_data(data): 
    """Function to plot spectral data with correct axis labels, y-units, and x-axis inversion."""
    print("What type of data are you using?")
    print("1. IR - Transmission")
    print("2. IR - Absorbance")
    print("3. Raman")
    print("4. UV-Vis")

    choice = input("Choose a data type (1-4): ")

    # Define labels, y-units, and x-axis inversion based on spectroscopy type
    if choice == '1':  # IR - Transmission
        xlabel = "Wavenumber (cm$^{-1}$)"
        ylabel = "Transmittance (%)"
        invert_x = True  # Invert x-axis for IR spectra
    elif choice == '2':  # IR - Absorbance
        xlabel = "Wavenumber (cm$^{-1}$)"
        ylabel = "Absorbance"
        invert_x = True
    elif choice == '3':  # Raman
        xlabel = "Raman Shift (cm$^{-1}$)"
        ylabel = "Intensity (counts)"
        invert_x = False
    elif choice == '4':  # UV-Vis
        xlabel = "Wavelength (nm)"
        ylabel = "Absorbance"
        invert_x = False
    else:
        print("Invalid choice. Using default settings.")
        xlabel = "X-axis"
        ylabel = "Y-axis"
        invert_x = False

    # Plot the spectral data
    ax = data.spc.transpose().plot(figsize=(10, 6))
    ax.set_title("Processed Spectral Data")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    

    # Apply x-axis inversion if required
    if invert_x:
        ax.invert_xaxis()

    plt.grid(True)
    plt.legend(fontsize="small")
    plt.show()


def plot_3d_data(data):
    """ Function to plot an interactive 3D plot using Plotly """
    df = data.spc.transpose()  # Transpose so wavenumbers are on the x-axis
    x = df.columns  # Wavenumbers
    y = df.index  # Spectra index
    z = df.values  # Intensity values

    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale="Viridis")])
    
    fig.update_layout(
        title="Interactive 3D Spectral Plot",
        scene=dict(
            xaxis_title="Wavenumber (cm^-1)",
            yaxis_title="Spectrum Index",
            zaxis_title="Intensity",
        )
    )

    fig.show()

def main():
    print("Welcome to the Spectral Data Processor!")
    
    # File input
    file_path = input("Enter the path to your spectral data file: ")
    
    try:
        data = SpectralData(file_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    processing_steps = []  # List to store selected preprocessing steps

    while True:
        print("\nSelect preprocessing steps (Choose multiple before applying):")
        print("1. Trim")
        print("2. Baseline Correction")
        print("3. Smoothing")
        print("4. Normalization")
        print("5. Centering")
        print("6. Derivative")
        print("7. Dataset Operations")
        print("8. Apply Selected Steps & Plot")
        print("9. Exit")

        choice = input("Choose an option (1-9): ")

        if choice == '1':
            print("1. Trim")
            print("2. Inverse Trim")
            sub_choice = input("Choose (1 or 2): ")
            start = float(input("Enter start value: "))
            end = float(input("Enter end value: "))
            processing_steps.append(("trim" if sub_choice == '1' else "invtrim", (start, end)))

        elif choice == '2':
            print("1. AsLS")
            print("2. Polyfit")
            print("3. Pearson")
            sub_choice = input("Choose (1-3): ")
            if sub_choice == '1':
                penalty = float(input("Penalty: "))
                asymmetry = float(input("Asymmetry: "))
                iterations = int(input("Iterations: "))
                processing_steps.append(("AsLS", (penalty, asymmetry, iterations)))
            elif sub_choice == '2':
                order = int(input("Polynomial order: "))
                iterations = int(input("Iterations: "))
                processing_steps.append(("polyfit", (order, iterations)))
            elif sub_choice == '3':
                u = int(input("U value: "))
                v = int(input("V value: "))
                processing_steps.append(("pearson", (u, v)))

        elif choice == '3':
            print("1. Rolling")
            print("2. Savitzky-Golay")
            sub_choice = input("Choose (1 or 2): ")
            if sub_choice == '1':
                window = int(input("Window size: "))
                processing_steps.append(("rolling", (window,)))
            elif sub_choice == '2':
                window = int(input("Window size: "))
                poly_order = int(input("Polynomial order: "))
                processing_steps.append(("SGSmooth", (window, poly_order)))

        elif choice == '4':
            print("1. SNV")
            print("2. MSC")
            print("3. Detrend")
            sub_choice = input("Choose (1-3): ")
            if sub_choice == '1':
                processing_steps.append(("snv", ()))
            elif sub_choice == '2':
                ref = int(input("Reference spectrum index (0-based): "))
                processing_steps.append(("msc", (ref,)))
            elif sub_choice == '3':
                order = int(input("Polynomial order: "))
                processing_steps.append(("detrend", (order,)))

        elif choice == '5':
            print("1. Mean Centering")
            print("2. Last Point Centering")
            sub_choice = input("Choose (1 or 2): ")
            if sub_choice == '1':
                self_val = int(input("Enter self value: "))
                processing_steps.append(("mean_center", (self_val,)))
            elif sub_choice == '2':
                processing_steps.append(("lastpoint", ()))

        elif choice == '6':
            print("1. SG Derivative")
            sub_choice = input("Choose (1): ")
            if sub_choice == '1':
                window = int(input("Window size: "))
                poly_order = int(input("Polynomial order: "))
                deriv_order = int(input("Derivative order: "))
                processing_steps.append(("SGDeriv", (window, poly_order, deriv_order)))

        elif choice == '7':
            print("1. Subtract")
            print("2. Reset")
            sub_choice = input("Choose (1 or 2): ")
            if sub_choice == '1':
                spectra_index = int(input("Spectra index: "))
                processing_steps.append(("subtract", (spectra_index,)))
            elif sub_choice == '2':
                processing_steps.append(("reset", ()))

        elif choice == '8':
            # Apply all selected preprocessing steps
            print("\nApplying selected preprocessing steps...\n")
            for method, params in processing_steps:
                if hasattr(data, method):
                    getattr(data, method)(*params)
                    print(f"Applied {method} with parameters {params}.")
                else:
                    print(f"Error: {method} is not a valid function in SpectralData.")

            # Let user choose between 2D and 3D plot
            while True:
                print("\nChoose a plotting method:")
                print("1. 2D Plot")
                print("2. 3D Plot")
                plot_choice = input("Enter choice (1 or 2): ")
                if plot_choice == '1':
                    plot_data(data)
                    break
                elif plot_choice == '2':
                    plot_3d_data(data)
                    break
                else:
                    print("Invalid choice. Please select 1 or 2.")

            processing_steps.clear()  # Reset steps after applying

        elif choice == '9':
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please select a valid option.")

if _name_ == "_main_":
    main()