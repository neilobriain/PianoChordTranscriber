"""
Utility script to plot an FFT graph of a chord file for use in report analysis.
"""

import librosa
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, fftfreq

FILEPATH = r"C:\Users\neilo\Desktop\chords\pianomajmin\G_maj_4_1.wav"

def plot_chord_fft(file_path):
    """
    Plot a normalised FFT graph of an audio file.
    """
    try:
        print("Preparing graph")
        
        # Load audio file
        samples, sampling_rate = librosa.load(file_path, sr=None, mono=True)
        n = len(samples)

        # Compute FFT and normalise amplitude spectrum
        signal_fft = fft(samples)
        amplitude_spectrum = np.abs(signal_fft)
        amplitude_spectrum /= np.max(amplitude_spectrum)  # Normalise to [0, 1]

        # Get frequency bins (Hz)
        freqs = fftfreq(n, 1 / sampling_rate)
        
        plt.plot(freqs[:2000], amplitude_spectrum[:2000])
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Normalised Amplitude")
        plt.title("Chord")
        plt.show()
        
        print("Success")
        
    except Exception as e:
        print(f"Cound not plot chord FFT: {e}")

plot_chord_fft(FILEPATH)