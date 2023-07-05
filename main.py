import numpy as np
import pandas as pd
from scipy.signal import medfilt, butter, lfilter, iirfilter
import matplotlib.pyplot as plt


# Median filter
def median(signal):
    array = np.array(signal)
    med_filtered = medfilt(array, kernel_size=3)
    return med_filtered


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a


# Butterworth filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


# Notch filter
def notch_filter(signal, fs, freq, ripple, order, filter_type):
    width = 1.0  # Width of the notch (adjust as needed)
    nyquist = 0.5 * fs
    low = (freq - width) / nyquist
    high = (freq + width) / nyquist
    b, a = iirfilter(order, [low, high], rp=ripple, btype='bandstop', analog=False, ftype=filter_type)
    filtered_signal = lfilter(b, a, signal)
    return filtered_signal


# High-pass filter
def highpass_filter(signal, lowcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    b, a = butter(order, low, btype='highpass')
    filtered_signal = lfilter(b, a, signal)
    return filtered_signal


# Low-pass filter
def lowpass_filter(signal, highcut, fs, order=5):
    nyquist = 0.5 * fs
    high = highcut / nyquist
    b, a = butter(order, high, btype='lowpass')
    filtered_signal = lfilter(b, a, signal)
    return filtered_signal


# Band-pass filter
def bandpass_filter(signal, fs, lowcut, highcut):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(4, [low, high], btype='bandpass')
    filtered_signal = lfilter(b, a, signal)
    return filtered_signal


# Load EEG signal from CSV file
def load_eeg_signal(filename):
    data = pd.read_csv(filename)
    print(data.columns)  # Print the column names to verify their correctness
    eeg_signal = data['RAW'].values
    return eeg_signal


"""""
In case of using dataset with defined electrodes in each column

def load_eeg_signal(filename):
    data = pd.read_csv(filename)
    print(data.columns)  # Print the column names to verify their correctness
    eeg_signal = data.values[:, :-1]
    return eeg_signal
    
"""""


# Generate image
def generate_image(signal, filtered_signal):
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(signal)
    plt.title('Original EEG signal')
    plt.subplot(2, 1, 2)
    plt.plot(filtered_signal)
    plt.title('Filtered EEG signal')
    plt.tight_layout()
    plt.savefig('eeg_signal_comparison.png')  # Save the image as 'eeg_signal_comparison.png'
    plt.show()


# Separation of frequency bands
def separate_frequency_bands(signal, fs):
    frequency_domain = np.fft.fft(signal)

    n = len(signal)
    frequency = np.fft.fftfreq(n, d=1.0 / fs)

    delta_band = (0.5, 4)
    theta_band = (4, 8)
    alpha_band = (8, 12)
    beta_band = (12, 30)
    gamma_band = (30, 100)

    delta_indices = np.where(np.logical_and(frequency >= delta_band[0], frequency < delta_band[1]))[0]
    theta_indices = np.where(np.logical_and(frequency >= theta_band[0], frequency < theta_band[1]))[0]
    alpha_indices = np.where(np.logical_and(frequency >= alpha_band[0], frequency < alpha_band[1]))[0]
    beta_indices = np.where(np.logical_and(frequency >= beta_band[0], frequency < beta_band[1]))[0]
    gamma_indices = np.where(np.logical_and(frequency >= gamma_band[0], frequency < gamma_band[1]))[0]

    delta_spectrum = np.abs(frequency_domain[delta_indices])
    theta_spectrum = np.abs(frequency_domain[theta_indices])
    alpha_spectrum = np.abs(frequency_domain[alpha_indices])
    beta_spectrum = np.abs(frequency_domain[beta_indices])
    gamma_spectrum = np.abs(frequency_domain[gamma_indices])

    return delta_spectrum, theta_spectrum, alpha_spectrum, beta_spectrum, gamma_spectrum


# Generate plots for frequency bands
def generate_combined_plots(signal, filtered_signal, fs):
    delta_spectrum, theta_spectrum, alpha_spectrum, beta_spectrum, gamma_spectrum = separate_frequency_bands(signal, fs)
    delta_filtered_spectrum, theta_filtered_spectrum, alpha_filtered_spectrum, beta_filtered_spectrum, gamma_filtered_spectrum = separate_frequency_bands(
        filtered_signal, fs)

    fig, axs = plt.subplots(5, 2, figsize=(12, 9))
    plt.subplots_adjust(bottom=0.2)

    # Delta band
    axs[0, 0].plot(delta_spectrum)
    axs[0, 0].set_title('Delta Band (Original)')
    axs[0, 1].plot(delta_filtered_spectrum)
    axs[0, 1].set_title('Delta Band (Filtered)')

    # Theta band
    axs[1, 0].plot(theta_spectrum)
    axs[1, 0].set_title('Theta Band (Original)')
    axs[1, 1].plot(theta_filtered_spectrum)
    axs[1, 1].set_title('Theta Band (Filtered)')

    # Alpha band
    axs[2, 0].plot(alpha_spectrum)
    axs[2, 0].set_title('Alpha Band (Original)')
    axs[2, 1].plot(alpha_filtered_spectrum)
    axs[2, 1].set_title('Alpha Band (Filtered)')

    # Beta band
    axs[3, 0].plot(beta_spectrum)
    axs[3, 0].set_title('Beta Band (Original)')
    axs[3, 1].plot(beta_filtered_spectrum)
    axs[3, 1].set_title('Beta Band (Filtered)')

    # Gamma band
    axs[4, 0].plot(gamma_spectrum)
    axs[4, 0].set_title('Gamma Band (Original)')
    axs[4, 1].plot(gamma_filtered_spectrum)
    axs[4, 1].set_title('Gamma Band (Filtered)')

    plt.tight_layout()
    plt.show()


def main():
    # Load EEG signal from file
    eeg_signal = load_eeg_signal('C:/Users/User-PC/Desktop/dps/raw4.csv')

    # User input for filter selection
    filter_choice = input(
        "Select filter (1 for Median, 2 for Butterworth, 3 for Notch, 4 for High-pass, 5 for Low-pass, 6 for Band-stop): ")

    if filter_choice == '1':
        # Apply median filter
        filtered_signal = median(eeg_signal)
    elif filter_choice == '2':
        # Apply Butterworth filter
        lowcut = 1.0  # Lower cutoff frequency
        highcut = 30.0  # Upper cutoff frequency
        fs = 512  # Sampling frequency (Hz)
        filtered_signal = butter_bandpass_filter(eeg_signal, lowcut, highcut, fs, order=5)
    elif filter_choice == '3':
        # Apply Notch filter
        fs = 512  # Sampling frequency (Hz)
        freq = 60.0  # Notch frequency to remove (e.g., power line interference)
        ripple = 5.0  # Passband ripple (dB)
        order = 5  # Filter order
        filter_type = 'cheby1'  # Filter type
        filtered_signal = notch_filter(eeg_signal, fs, freq, ripple, order, filter_type)
    elif filter_choice == '4':
        # Apply High-pass filter
        lowcut = 0.5  # Lower cutoff frequency
        fs = 512  # Sampling frequency (Hz)
        filtered_signal = highpass_filter(eeg_signal, lowcut, fs, order=5)
    elif filter_choice == '5':
        # Apply Low-pass filter
        highcut = 30.0  # Upper cutoff frequency
        fs = 512  # Sampling frequency (Hz)
        filtered_signal = lowpass_filter(eeg_signal, highcut, fs, order=5)
    elif filter_choice == '6':
        # Apply Bandpass filter
        lowcut = 10  # Lower cutoff frequency
        highcut = 20  # Upper cutoff frequency
        fs = 512  # Sampling frequency (Hz)
        filtered_signal = bandpass_filter(eeg_signal, fs, lowcut, highcut)

    else:
        print("Invalid filter choice.")
        return

    # Generate image showing the original and filtered EEG signals
    generate_image(eeg_signal, filtered_signal)
    fs = 512  # Define the default sampling frequency (Hz)
    # Generate frequency band plots
    generate_combined_plots(eeg_signal, filtered_signal, fs)


if __name__ == "__main__":
    main()
