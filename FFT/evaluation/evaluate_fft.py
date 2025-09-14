"""
Evalute FFT method's accuracy by logging statistics of its accuracy,
as well as creating confusion matrix and class accuracy CSV files.
"""

import os
import sys
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm
import csv

# Add parent directory to the module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fft import get_chord_name

dataset_paths = {'GTR3': r'D:\kaggle\GTR3\AllChords Evaluation',
            'PTW': r'D:\kaggle\Piano Triads Waveset\piano_triads_Evaluation',
            'aPTD': r'D:\datasets\audioPianoTriadDataset\audioPianoTriadDataset\Chords\MajMin'}

DATASET = "GTR3"
folder_path = Path(dataset_paths[DATASET])

log_file = Path(f"{DATASET} FFT evaluation_log.txt")
class_accuracy_csv = Path(f"{DATASET} FFT class_accuracy.csv")
confusion_csv = Path(f"{DATASET} FFT confusion_pairs.csv")

# Counters
total_checked = 0
correct = 0
root_correct = 0
strong_note_correct = 0
incorrect = 0

confusion_counter = Counter()
mistakes_counter = Counter()
class_totals = defaultdict(int)
class_correct = defaultdict(int)

fifth_correct = 0
relative_major_minor_correct = 0

# Matching helpers
def is_chord_match(test_chord, prediction):
    return test_chord == prediction[0]

def is_root_match(test_chord, prediction):
    return test_chord[0] == prediction[0][0]

def is_strong_note_match(test_chord, prediction):
    return test_chord[0] == prediction[3]

def note_to_int(note):
    mapping = {'C':0, 'C#':1, 'Db':1, 'D':2, 'D#':3, 'Eb':3, 'E':4,
               'F':5, 'F#':6, 'Gb':6, 'G':7, 'G#':8, 'Ab':8,
               'A':9, 'A#':10, 'Bb':10, 'B':11}
    return mapping[note]

def is_fifth_apart(root1, root2):
    return (note_to_int(root1) - note_to_int(root2)) % 12 in (5, 7)

def is_relative_major_minor(chord1, chord2):
    # Example: Cmaj vs Am
    if chord1.endswith("maj") and chord2.endswith("min"):
        return (note_to_int(chord2[0]) - note_to_int(chord1[0])) % 12 == 9
    if chord1.endswith("min") and chord2.endswith("maj"):
        return (note_to_int(chord1[0]) - note_to_int(chord2[0])) % 12 == 9
    return False

# Gather all .wav files first
wav_files = []
for root, _, files in os.walk(folder_path):
    for filename in files:
        if filename.lower().endswith(".wav"):
            wav_files.append((Path(root) / filename, Path(root).name))

# Process files
with log_file.open("w", encoding="utf-8") as log:
    
    log.write(f"{DATASET} dataset FFT analysis.\n\n")
    log.write("File name | Correct chord | Prediction info (chord prediction, threshold, detected notes, strong note)\n\n")
    
    for filepath, folder_name in tqdm(wav_files, desc="Evaluating Chord Predictions", unit="file"):
        try:
            total_checked += 1
            prediction = get_chord_name(filepath)
            filename = Path(filepath).name
            
            # Log each prediction
            log.write(f"{filename} {folder_name} {prediction}\n")

            # Per-class tracking
            class_totals[folder_name] += 1
            
            # Track predictions for confusion matrix
            confusion_counter[(folder_name, prediction[0])] += 1

            if is_chord_match(folder_name, prediction):
                correct += 1
                class_correct[folder_name] += 1
            else:
                incorrect += 1
                mistakes_counter[(folder_name, prediction[0])] += 1

                if is_root_match(folder_name, prediction):
                    root_correct += 1
                if is_strong_note_match(folder_name, prediction):
                    strong_note_correct += 1
                if is_fifth_apart(folder_name[0], prediction[0][0]):
                    fifth_correct += 1
                if is_relative_major_minor(folder_name, prediction[0]):
                    relative_major_minor_correct += 1

        except Exception as e:
            log.write(f"Failed to process {Path(filepath).name}: {e}\n")

    # Summary
    log.write("\nCompleted\n")
    log.write(f"Total chords checked: {total_checked}\n")
    log.write(f"Correct: {correct} ({correct/total_checked*100:.2f}%)\n")
    log.write(f"Incorrect: {incorrect} ({incorrect/total_checked*100:.2f}%)\n")
    log.write(f"Incorrect but correct root: {root_correct} ({root_correct/incorrect*100 if incorrect else 0:.2f}%)\n")
    log.write(f"Incorrect but correct strong note: {strong_note_correct} ({strong_note_correct/incorrect*100 if incorrect else 0:.2f}%)\n")
    log.write(f"Incorrect but root a perfect fifth apart: {fifth_correct} ({fifth_correct/incorrect*100 if incorrect else 0:.2f}%)\n")
    log.write(f"Incorrect but relative major/minor: {relative_major_minor_correct} ({relative_major_minor_correct/incorrect*100 if incorrect else 0:.2f}%)\n")

    if total_checked != (correct + incorrect):
        log.write("Warning: correct/incorrect count does not equal total checked\n")

    # Top mistakes
    log.write("\nTop 10 most common mistakes:\n")
    for (true_chord, pred_chord), count in mistakes_counter.most_common(10):
        log.write(f"{true_chord} → {pred_chord}: {count}\n")

    # Per-class accuracy
    log.write("\nPer-class accuracy:\n")
    for chord in sorted(class_totals):
        acc = class_correct[chord] / class_totals[chord] * 100
        log.write(f"{chord}: {acc:.2f}% ({class_correct[chord]}/{class_totals[chord]})\n")

# Save per-class accuracy to CSV
with class_accuracy_csv.open("w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Chord", "Correct", "Total", "Accuracy (%)"])
    for chord in sorted(class_totals):
        acc = class_correct[chord] / class_totals[chord] * 100
        writer.writerow([chord, class_correct[chord], class_totals[chord], f"{acc:.2f}"])

# Save confusion matrix in proper matrix format
all_chords = sorted(set(class_totals.keys()))
with confusion_csv.open("w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    # Header row
    writer.writerow(["True/Predicted"] + all_chords)
    # Data rows
    for true_chord in all_chords:
        row = [true_chord]
        for pred_chord in all_chords:
            count = confusion_counter.get((true_chord, pred_chord), 0)
            row.append(count)
        writer.writerow(row)