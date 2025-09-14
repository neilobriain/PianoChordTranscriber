"""
Utility script to run all WAV files in a selected folder through the
CNN model and print its predictions and confidence scores to console.
"""

import os
import sys

# Add parent directory to the module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cnn_model import predict_chord

folder_path = 'C:/Users/neilo/Desktop/chords/pianotriad'

# Loop over WAV files and predict chords
for filename in os.listdir(folder_path):
    if filename.lower().endswith(".wav"):
        filepath = os.path.join(folder_path, filename)

        try:
            print(filename, predict_chord(filepath))

        except Exception as e:
            print(f"Failed to process {filename}: {e}")