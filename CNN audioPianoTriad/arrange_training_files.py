"""
Organises files from the audioPianoTriad (Augmented x 10) dataset into chord-specific folders, which
will be later used as chord name labels during CNN training.

Expected filename format:
    piano_<instrumentID>_<rootCode>_<triadCode>_<extra>_<index>.wav
Example:
    piano_4_Af_d_m_45.wav  → rootCode="Af", triadCode="d" → Abdim/
"""

import os
import shutil
from tqdm import tqdm

# Folder containing the .wav files (augmented x10 version)
base_dir = r"D:\datasets\audioPianoTriadDataset\audioPianoTriadDataset\Chords"

# Mapping for root notes
note_map = {
    'Cn': 'C',  'Df': 'Db', 'Dn': 'D',  'Ef': 'Eb',
    'En': 'E',  'Fn': 'F',  'Gf': 'Gb', 'Gn': 'G',
    'Af': 'Ab', 'An': 'A',  'Bf': 'Bb', 'Bn': 'B'
}

# Mapping for triad types
triad_map = {
    'j': 'maj',
    'n': 'min',
    'a': 'aug',
    'd': 'dim'
}

# Get all .wav files in the directory
files = [f for f in os.listdir(base_dir) if f.endswith('.wav')]

# Process files with a progress bar
for filename in tqdm(files, desc="Arranging chord files"):
    try:
        # Metadata is in the file name: piano_4_Af_d_m_45.wav
        parts = filename.split('_')
        if len(parts) < 6:
            continue  # skip malformed names

        root = parts[2]  # e.g., "Af"
        triad_code = parts[3]  # e.g., "d"

        # Map to final names
        root_name = note_map.get(root)
        triad_name = triad_map.get(triad_code)

        if not root_name or not triad_name:
            continue  # skip if invalid code

        target_folder = os.path.join(base_dir, f"{root_name}{triad_name}")
        os.makedirs(target_folder, exist_ok=True)

        src_path = os.path.join(base_dir, filename)
        dst_path = os.path.join(target_folder, filename)

        shutil.move(src_path, dst_path)
    
    except Exception as e:
        print(f"Error processing {filename}: {e}")