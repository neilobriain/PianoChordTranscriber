"""
Utility script to run all WAV files in a selected folder through the
FFT model and print its predictions and information to console.
"""

import os
import sys

# Add parent directory to the module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fft import get_chord_name

folder_path = 'C:/Users/neilo/Desktop/chords/pianotriad'

# Loop over WAV files and predict chords
for filename in os.listdir(folder_path):
    if filename.lower().endswith(".wav"):
        filepath = os.path.join(folder_path, filename)

        try:
            print(filename, get_chord_name(filepath))

        except Exception as e:
            print(f"Failed to process {filename}: {e}")