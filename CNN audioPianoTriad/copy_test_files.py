"""
Randomly selects and copies a fixed number of piano chord files 
from each chord folder into a single destination directory.

This utility is designed for the audioPianoTriads dataset and to retrieve a
number of examples of each chord type. These are saved into a single test folder
that can then be used as a source for quick testing of prediction functions more generally,
outside of model training.
"""

import shutil
import random
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)

# Source directory containing chord subfolders
SOURCE_DIR = r'D:\datasets\audioPianoTriadDataset\audioPianoTriadDataset\Chords\MajMin'

# Desktop path - adjust if needed for your system
DESKTOP_PATH = Path.home() / 'Desktop'

# Destination directory structure
DEST_BASE = DESKTOP_PATH / 'chords'
DEST_DIR = DEST_BASE / 'pianotriad'

# Number of files to copy from each chord folder
FILES_PER_CHORD = 3

def create_destination_folder():
    """Create the destination folder structure"""
    try:
        DEST_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Created destination folder: {DEST_DIR}")
        return True
    except Exception as e:
        print(f"Error creating destination folder: {e}")
        return False

def get_chord_folders(source_dir):
    """Get all chord subfolders from the source directory"""
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"Source directory does not exist: {source_dir}")
        return []
    
    chord_folders = [folder for folder in source_path.iterdir() 
                    if folder.is_dir()]
    
    print(f"Found {len(chord_folders)} chord folders:")
    for folder in chord_folders:
        print(f"  - {folder.name}")
    
    return chord_folders

def get_wav_files(folder_path):
    """Get all .wav files from a folder"""
    wav_files = list(folder_path.glob('*.wav'))
    return wav_files

def copy_random_files(chord_folders, dest_dir, files_per_chord):
    """Copy random files from each chord folder to destination"""
    total_copied = 0
    copy_log = []
    
    for chord_folder in chord_folders:
        chord_name = chord_folder.name
        print(f"\nProcessing chord: {chord_name}")
        
        # Get all wav files in this chord folder
        wav_files = get_wav_files(chord_folder)
        
        if not wav_files:
            print(f"  No .wav files found in {chord_name}")
            continue
            
        print(f"  Found {len(wav_files)} .wav files")
        
        # Select random files (or all files if fewer than requested)
        num_to_copy = min(files_per_chord, len(wav_files))
        selected_files = random.sample(wav_files, num_to_copy)
        
        print(f"  Copying {num_to_copy} random files...")
        
        # Copy selected files
        for i, file_path in enumerate(selected_files, 1):
            try:
                # Create new filename with chord prefix to avoid naming conflicts
                new_filename = f"{chord_name}_{file_path.name}"
                dest_file_path = dest_dir / new_filename
                
                # Copy the file
                shutil.copy2(file_path, dest_file_path)
                
                print(f"    {i}/{num_to_copy}: {file_path.name} -> {new_filename}")
                copy_log.append(f"{chord_name}: {file_path.name} -> {new_filename}")
                total_copied += 1
                
            except Exception as e:
                print(f"    Error copying {file_path.name}: {e}")
    
    return total_copied, copy_log

def main():
    print("=== Piano Chord WAV File Copier ===")
    print(f"Source: {SOURCE_DIR}")
    print(f"Destination: {DEST_DIR}")
    print(f"Files per chord: {FILES_PER_CHORD}")
    print("=" * 50)
    
    # Create destination folder
    if not create_destination_folder():
        return
    
    # Get chord folders
    chord_folders = get_chord_folders(SOURCE_DIR)
    if not chord_folders:
        print("No chord folders found. Exiting.")
        return
    
    # Copy random files
    total_copied, copy_log = copy_random_files(chord_folders, DEST_DIR, FILES_PER_CHORD)
    
    # Summary
    print("\n" + "=" * 50)
    print(f"SUMMARY:")
    print(f"Total chords processed: {len(chord_folders)}")
    print(f"Total files copied: {total_copied}")
    print(f"Destination: {DEST_DIR}")
    
    # Save copy log to file
    log_file = DEST_DIR / 'copy_log.txt'
    try:
        with open(log_file, 'w') as f:
            f.write("Piano Chord WAV Copy Log\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Source: {SOURCE_DIR}\n")
            f.write(f"Files per chord: {FILES_PER_CHORD}\n")
            f.write(f"Total files copied: {total_copied}\n\n")
            f.write("Copied files:\n")
            f.write("-" * 20 + "\n")
            for entry in copy_log:
                f.write(f"{entry}\n")
        
        print(f"Copy log saved to: {log_file}")
    except Exception as e:
        print(f"Could not save copy log: {e}")
    
    print("\nOperation completed successfully!")

if __name__ == "__main__":
    main()