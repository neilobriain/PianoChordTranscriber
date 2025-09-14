"""
Evalute FFT method's accuracy by logging statistics of its accuracy. An improved
version of this script (evaluate_fft.py) is used for final report statistics.
"""

import os
import sys
from pathlib import Path
from tqdm import tqdm

# Add parent directory to the module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fft import get_chord_name

folder_path = Path(r'D:\kaggle\GTR3\AllChords Evaluation')
# folder_path = Path('D:\kaggle\Piano Triads Waveset\piano_triads_Evaluation')
# folder_path = Path(r'D:\datasets\audioPianoTriadDataset\audioPianoTriadDataset\Chords\MajMin')
log_file = Path("evaluation_log.txt")

total_checked = 0
correct = 0
root_correct = 0
strong_note_correct = 0
incorrect = 0

def is_chord_match(test_chord, prediction):
    return test_chord == prediction[0]
    
def is_root_match(test_chord, prediction):
    return test_chord[0] == prediction[0][0]
    
def is_strong_note_match(test_chord, prediction):
    return test_chord[0] == prediction[3]

# Gather all .wav files first so tqdm knows the total
wav_files = []
for root, _, files in os.walk(folder_path):
    for filename in files:
        if filename.lower().endswith(".wav"):
            wav_files.append((Path(root) / filename, Path(root).name))

# Process files with progress bar
with log_file.open("w", encoding="utf-8") as log:
    
    log.write("Filename | Correct Chord | Prediction Info\n\n")
    
    for filepath, folder_name in tqdm(wav_files, desc="Evaluating Chord Predictions", unit="file"):
        try:
            total_checked += 1
            prediction = get_chord_name(filepath)

            filename = Path(filepath).name
            
            # Log details
            log.write(f"{filename} {folder_name} {prediction}\n")

            if is_chord_match(folder_name, prediction):
                correct += 1
            else:
                incorrect += 1
                if is_root_match(folder_name, prediction):
                    root_correct += 1
                if is_strong_note_match(folder_name, prediction):
                    strong_note_correct += 1

        except Exception as e:
            log.write(f"Failed to process {filepath}: {e}\n")

    # Write summary
    log.write("\nCompleted\n")
    log.write(f"Total chords checked: {total_checked}\n")
    log.write(f"Correct: {correct}, Incorrect: {incorrect}\n")
    log.write(f"Incorrect, but correct root prediction: {root_correct}\n")
    log.write(f"Incorrect, but correct strong note prediction: {strong_note_correct}\n")

    if total_checked != (correct + incorrect):
        log.write("Warning: correct/incorrect count does not equal total checked\n")