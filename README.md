# EEG signal processing
This project provides a collection of functions for processing EEG (electroencephalogram) signals. It includes various filters for noise reduction and frequency band separation, as well as functionality for generating visualizations of the processed signals.

# Instalation
To use this project, you need to have the following dependencies installed:
- NumPy
- pandas
- SciPy
- Matplotlib
You can install these dependencies by running the following command:
pip install numpy pandas scipy matplotlib

# Usage
The main functionality of this project is contained in the main function, which performs the processing of the EEG signal.
Upon running the program, you will be prompted to select a filter to apply to the EEG signal. You can choose from the following options:
- Median filter
- Butterworth filter
- Notch filter
- High-pass filter
- Low-pass filter
- Band-stop filter

After selecting a filter, the program will process the EEG signal accordingly and generate an image comparing the original and filtered signals. Additionally, it will generate plots showing the amplitude spectra for different frequency bands.

# Author
- Emina Efendić
