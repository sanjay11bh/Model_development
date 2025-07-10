import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from PIL import ImageTk, Image
import os
import shutil
from spectral import SpectralData
import numpy as np

class SpectralProcessorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Spectral Processor GUI")
        self.root.geometry("800x600")
        self.root.iconbitmap(r'C:\Users\vaibh\OneDrive\Desktop\model_delopment\utils\logo-wide-white.ico')
        self.img = ImageTk.PhotoImage(Image.open(r'C:\Users\vaibh\OneDrive\Desktop\model_delopment\utils\image.png'))
        self.image_label = tk.Label(root, image=self.img)
        self.image_label.pack()

        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.axis = self.fig.add_subplot(111)

        self.file_path = None
        self.userData = None

        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill='both', expand=True)

        self.left_frame = tk.Frame(self.main_frame)
        self.left_frame.pack(side='left', fill='both', expand=True, padx=10, pady=10)

        self.middle_frame = tk.Frame(self.main_frame)
        self.middle_frame.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        
        self.middle_frame_top = tk.Frame(self.middle_frame)
        self.middle_frame_top.pack(side='top', fill='both', expand=True, padx=10, pady=10)

        self.middle_frame_bottom = tk.Frame(self.middle_frame)
        self.middle_frame_bottom.pack(side='bottom', fill='both', expand=True, padx=10, pady=10)

        self.right_frame = tk.Frame(self.main_frame)
        self.right_frame.pack(side='right', fill='both', expand=True, padx=10, pady=10)

        # Spectroscopy Type
        self.spectroscopy_type_label = ttk.Label(self.left_frame, text="Select Spectroscopy Type:")
        self.spectroscopy_type_label.pack(anchor="w", pady=5)

        self.spectroscopy_options = {
            "IR - Transmission": ("Wavenumber (cm$^{-1}$)", "Transmittance (%)", True),
            "IR - Absorbance": ("Wavenumber (cm$^{-1}$)", "Absorbance", True),
            "Raman": ("Raman Shift (cm$^{-1}$)", "Intensity (counts)", False),
            "UV-Vis": ("Wavelength (nm)", "Absorbance", False)
        }

        self.spec_type = ttk.Combobox(self.left_frame, values=list(self.spectroscopy_options))
        self.spec_type.pack(anchor="w", pady=0.1, padx=1)

        # File Loading
        self.load_button = ttk.Button(self.left_frame, text="Load Spectral Data", command=self.load_spectral_data)
        self.load_button.pack(anchor="w", pady=0.1, padx=1)

        self.path_value = tk.Entry(self.left_frame, font=4)
        self.path_value.pack(anchor="w", pady=0.1, padx=1)

        # Preprocessing
        self.preprocessing_label = ttk.Label(self.left_frame, text="Select Preprocessing Step:")
        self.preprocessing_label.pack(anchor="w", pady=5)

        self.preprocessing_dict = {
            'Trim': ['Trim', 'Inverse Trim'],
            'Baseline Correction': ['AsLS', 'Polyfit', 'Pearson'],
            'Smoothing': ['Rolling', 'Savitzky-Golay'],
            'Normalization': ['SNV', 'Detrend', 'MSC', 'Area', 'Peak Normalization', 'Vector', 'Min-max', 'Pareto'],
            'Center': ['Mean', 'Last Point'],
            'Derivative': ['SG Derivative'],
            'Dataset': ['Subtract', 'Reset'],
            '': ''
        }

        self.labels = {
            "Trim": ["Start", "End"],
            "Inverse Trim": ['Start', 'End'],
            "AsLS": ["Penalty", "Asymmetry", "Iterations"],
            "Polyfit": ["Order", "Iterations"],
            "Rolling": ["Window"],
            "Savitzky-Golay": ["Window", "Poly. Order"],
            'SNV': [],
            'MSC': ["Reference"],
            'Area': [],
            'Peak Normalization': ["Peak position"],
            'Vector': [],
            'Min-max': ['Min', 'Max'],
            'Mean': ['Self'],
            'Last Point': [],
            'SG Derivative': ['Window', 'Polynomial', 'Deriv. Order'],
            'Reset': [],
            'Subtract': ['Spectrum'],
            'Detrend': ['Order'],
            'Pareto': [],
            'Pearson': ['u', 'v'],
            '': []
        }
        
        self.method_name_map = {
                'AsLS': 'AsLS',
                'Polyfit': 'polyfit',
                'Rolling': 'rolling',
                'Savitzky-Golay': 'SGSmooth',
                'SNV': 'snv',
                'MSC': 'msc',
                'Area': 'area',
                'Peak Normalization': 'peaknorm',
                'Vector': 'vector',
                'Min-max': 'minmax',
                'Mean': 'mean_center',
                'Last Point': 'lastpoint',
                'SG Derivative': 'SGDeriv',
                'Detrend': 'detrend',
                'Pareto': 'pareto',
                'Pearson': 'pearson',
                'Reset': 'reset',
                'Subtract': 'subtract',
                'Trim': 'trim',
                'Inverse Trim': 'invtrim',
            }

        ttk.Label(self.left_frame, text='Category').pack(anchor="w", pady=0.1, padx=1)
        self.preprocessing_steps = ttk.Combobox(self.left_frame, state='readonly')
        self.preprocessing_steps['values'] = tuple(self.preprocessing_dict.keys())
        self.preprocessing_steps.pack(anchor="w", pady=0.1, padx=1)
        self.preprocessing_steps.bind("<<ComboboxSelected>>", self.update_methods)

        ttk.Label(self.left_frame, text='Technique').pack(anchor="w", pady=0.1, padx=1)
        self.method = ttk.Combobox(self.left_frame, state='readonly')
        self.method.pack(anchor="w", pady=0.1, padx=1)
        self.method.bind("<<ComboboxSelected>>", self.update_labels)

        self.opt_labels = []
        self.options = []

        for _ in range(4):
            label = ttk.Label(self.left_frame, text='', width=12)
            entry = ttk.Entry(self.left_frame, width=12)
            label.pack(anchor="w", pady=0.1, padx=1)
            entry.pack(anchor="w", pady=0.1, padx=1)
            label.pack_forget()
            entry.pack_forget()
            entry.config(state='disabled')
            self.opt_labels.append(label)
            self.options.append(entry)
        

        self.sub_button_1_frame = ttk.Frame(self.left_frame)
        self.sub_button_1_frame.pack(anchor="w", pady=5)

        self.fourier_button = ttk.Button(self.sub_button_1_frame, text="+ Fourier Transform", command = self.Fourier_transform)
        self.fourier_button.pack(side="left", pady=5)

        self.wavelet_button = ttk.Button(self.sub_button_1_frame, text="+ Wavelet Transform", command = self.Wavelet_transform)
        self.wavelet_button.pack(side="left", pady=5)

        self.apply_button = ttk.Button(self.sub_button_1_frame, text="Apply Preprocessing", command=self.get_pp_preprocessing_steps)
        self.apply_button.pack(side="left", pady=5)

        
        self.step_display = tk.Text(self.left_frame, height=8, width=40)
        self.step_display.pack(anchor="w", pady=5)
        
        self.button_frame = ttk.Frame(self.left_frame)
        self.button_frame.pack(anchor="w", pady=5)

        self.save_param_button = ttk.Button(self.button_frame, text="Save log", command= self.save_param_logs)
        self.save_param_button.pack(side="left", pady=5)
 
        self.reset_button = ttk.Button(self.button_frame, text="Reset", command=self.reset_all)
        self.reset_button.pack(side="left", pady=5)
        
        self.save_processed_spectrum = ttk.Button(self.button_frame, text = "Save processed spectrum", command = self.save_processed_spectrum)
        self.save_processed_spectrum.pack(side="left", pady=5)

        self.canvas_top = FigureCanvasTkAgg(self.fig, master=self.middle_frame_top)
        self.canvas_top.draw()
        self.canvas_top.get_tk_widget().pack(fill='both', expand=True)

        self.toolbar_top = NavigationToolbar2Tk(self.canvas_top, self.middle_frame_top)
        self.toolbar_top.update()
        self.toolbar_top.pack()
        
        self.fig_bottom = Figure(figsize=(5, 4), dpi=100)
        self.axis_bottom = self.fig_bottom.add_subplot(111)

        self.canvas_bottom = FigureCanvasTkAgg(self.fig_bottom, master=self.middle_frame_bottom)
        self.canvas_bottom.draw()
        self.canvas_bottom.get_tk_widget().pack(fill='both', expand=True)

        self.toolbar_bottom = NavigationToolbar2Tk(self.canvas_bottom, self.middle_frame_bottom)
        self.toolbar_bottom.update()
        self.toolbar_bottom.pack()

        self.model_label = ttk.Label(self.right_frame, text="Select model:")
        self.model_label.pack(anchor="w", pady=5)

        self.models = [
            "PLS", "XGBoost", "SVM", "RF", "Decision Tree", "LightG", "DNW", "ADABoost", "GB", "Gaussian", "Lasso"
        ]

        self.models_combobox = ttk.Combobox(self.right_frame, values=self.models)
        self.models_combobox.pack(anchor="w", pady=0.1, padx=1)

    def load_spectral_data(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("All Files", "*.*"), ("CSV Files", "*.csv")])
        if not self.file_path:
            return
        try:
            self.path_value.delete(0, tk.END)
            self.path_value.insert(0, self.file_path)
            file_dir = os.path.dirname(self.file_path)
            file_base = os.path.splitext(os.path.basename(self.file_path))[0]
            output_dir = os.path.join(file_dir, f"{file_base}_y=mx+c_folder")
            os.makedirs(output_dir, exist_ok=True)
            raw_file_copy_path = os.path.join(output_dir, os.path.basename(self.file_path))
            shutil.copy(self.file_path, raw_file_copy_path)
            self.userData = SpectralData(self.file_path)
            self.plot_data()
            messagebox.showinfo("Success", "File loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")

    def update_methods(self, event):
        selected_category = self.preprocessing_steps.get()
        self.method['values'] = tuple(self.preprocessing_dict.get(selected_category, []))
        self.method.set('')
        self.update_labels(None)

    def update_labels(self, event):
        for label, option in zip(self.opt_labels, self.options):
            label.pack_forget()
            option.pack_forget()
            option.config(state='disabled')
            label.config(text='')

        selected_method = self.method.get()
        if selected_method in self.labels:
            for i, label_text in enumerate(self.labels[selected_method]):
                self.opt_labels[i].config(text=label_text)
                self.opt_labels[i].pack(anchor="w", pady=0.1, padx=1)
                self.options[i].pack(anchor="w", pady=0.1, padx=1)
                self.options[i].config(state='normal')
    


    def Wavelet_transform(self):
        
        self.step_display.insert(tk.END, "Wavelet Transform applied\n")
    
    
        return 
    




    def Fourier_transform(self):
        """
        fourier transforms the data 
        """
        df = self.userData.spc  
        import scipy.fftpack as fft 

        if df.shape[0] < df.shape[1]:
            df = df.T

        f = fft.fft(df, axis=1)  
        fhat = np.abs(f)

        self.step_display.insert(tk.END, "Fourier Transform applied\n")

        self.axis.clear()
        self.axis.plot(fhat)  # Transpose to plot all spectra in one go
        self.axis.set_xlabel("Frequency (a.u.)", fontweight="bold", fontsize=16)
        self.axis.set_ylabel("Intensity (a.u.)", fontweight="bold", fontsize=16)
        self.axis.set_title("Fourier Transform", fontweight="bold", fontsize=16)
        self.canvas_top.draw()

    def get_pp_preprocessing_steps(self):
        category = self.preprocessing_steps.get()
        method_key = self.method.get()
        method_name = self.method_name_map.get(method_key, method_key.lower())
        param = []

        for option in self.options:
            if option.winfo_ismapped() and option.get().strip() != '':
                try:
                    val = float(option.get())
                except ValueError:
                    val = option.get()
                param.append(val)

        try:
            if hasattr(self.userData, method_name):
                func = getattr(self.userData, method_name)
                func(*param)
                self.step_display.insert(tk.END, f"{method_key}({', '.join(map(str, param))})\n")

                self.plot_data()
            else:
                messagebox.showerror("Method Error", f"No such method: {method_key}")
        except Exception as e:
            messagebox.showerror("Processing Error", f"Failed to apply method {method_key}: {e}")
           

    def plot_data(self):
        self.axis.clear()
        self.axis.plot(self.userData.spc.transpose())

        if self.userData.spc.shape[0] < 10:
            self.axis.legend([str(x + 1) for x in range(self.userData.spc.shape[0])], loc='best', ncol=1)
        elif self.userData.spc.shape[0] < 20:
            self.axis.legend([str(x + 1) for x in range(self.userData.spc.shape[0])], loc='best', ncol=2)

        labels = self.axis.get_xticklabels() + self.axis.get_yticklabels()
        for label in labels:
            label.set_fontweight('bold')

        spec_type = self.spec_type.get()
        prefix = ''
        invert_x = False

        if spec_type in self.spectroscopy_options:
            xlabel, ylabel_with_units, invert_x = self.spectroscopy_options[spec_type]
            if invert_x:
                self.axis.invert_xaxis()
            self.axis.set_xlabel(xlabel, fontweight="bold", fontsize=16)
            self.axis.set_ylabel(f"{prefix} {ylabel_with_units}", fontweight="bold", fontsize=16)
        else:
            self.axis.set_xlabel("Energy", fontweight="bold", fontsize=16)
            self.axis.set_ylabel("Intensity (a.u.)", fontweight="bold", fontsize=16)

        self.canvas_top.draw()
    
    def save_param_logs(self):
        try:
            log_text = self.step_display.get("1.0", tk.END).strip()
            if not log_text:
                messagebox.showwarning("Empty Log", "No preprocessing steps to save.")
                return

            file_dir = os.path.dirname(self.file_path)
            file_base = os.path.splitext(os.path.basename(self.file_path))[0]
            output_dir = os.path.join(file_dir, f"{file_base}_y=mx+c_folder")
            os.makedirs(output_dir, exist_ok=True)

            output_file = os.path.join(output_dir, f"{file_base}_preprocessing_log.txt")
            with open(output_file, "w") as f:
                f.write(log_text)

            messagebox.showinfo("Success", f"Preprocessing log saved at:\n{output_file}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save preprocessing log: {e}")
    
    def reset_all(self):
        self.spec_type.set('')
        self.path_value.delete(0, tk.END)
        self.userData = None
        self.fig.clear()
        self.axis = self.fig.add_subplot(111)
        self.canvas_top.draw()
        self.step_display.delete("1.0", tk.END)

        for label, option in zip(self.opt_labels, self.options):
            label.pack_forget()
            option.pack_forget()
            option.config(state='disabled')
            label.config(text='')
            option.delete(0, tk.END)

        messagebox.showinfo("Reset", "All fields have been reset.")
   
    def save_processed_spectrum(self):
        file_dir = os.path.dirname(self.file_path)
        file_base = os.path.splitext(os.path.basename(self.file_path))[0]
        output_dir = os.path.join(file_dir, f"{file_base}_y=mx+c_folder")
        if self.userData is None:
            messagebox.showwarning("❌ No Data present ❌")
            return 
        
        fpath = filedialog.asksaveasfilename(defaultextension=".csv",initialdir= output_dir,filetypes=[("All Files", "*.*"), ("CSV Files", "*.csv")])

        if not fpath:
            return
        
        try:
            if fpath.lower().endswith('.csv'):
                self.userData.export_csv(fpath)
            else:
                self.userData.export_excel(fpath)
            
            messagebox.showinfo("Success", f"Processed spectrum saved at:\n{fpath}")

        except Exception as e:
            messagebox.showerror("Error", f"Could not save processed spectrum: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SpectralProcessorGUI(root)
    root.mainloop()
     