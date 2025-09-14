"""
This module provides an FFT-based method for transcribing
the chord played in an audio file. This is done through the
get_chord_name() function along with the helper functions found
in this module.
"""

import librosa
import numpy as np
from numpy.fft import fft, fftfreq

# Don't try to predict if more notes than this found in chord
NOTE_NUM_LIMIT = 5

# Mapping from note index to note name
note_names = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

def frequency_to_note_name(frequency):
    """
    Helper function for get_chord_name. Returns MIDI note number and note name calculated from its frequency Hz.
    """
    note_number = 12 * np.log2(frequency / 440.0) + 69
    note_number = int(round(note_number))
    note_index = note_number % 12
    return note_number, note_names[note_index]

def identify_chord(notes):
    """
    Helper function for get_chord_name. Iterates through chord dictionary to find a match.
    """
    # Get unique pitch classes modulo 12 (remove duplicates and octave info)
    unique_notes = sorted(set(n[0] % 12 for n in notes))
    
    # All possible chords to check, as sets of intervals
    chord_intervals = {
        'maj': {0, 4, 7},
        'min': {0, 3, 7},
        'sus2': {0, 2, 7},
        'sus4': {0, 5, 7},
        'dim': {0, 3, 6},
        'aug': {0, 4, 8}
    }
    
    # Try each note as root and check if triad matches
    for root in unique_notes:
        # Transpose notes relative to root
        transposed = {(note - root) % 12 for note in unique_notes}
        
        # If no of notes above limit, return 'L' to signify threshold being too low
        if len(transposed) > NOTE_NUM_LIMIT:
            return 'L'
        
        # Check against chord dictionary
        for chord_type, intervals in chord_intervals.items():
            if intervals.issubset(transposed):
                return f"{note_names[root]}{chord_type}"

    # If no chord match found
    return 'Unknown Chord'

def get_chord_name(file_path):
    """
    Uses the FFT transcriber method to guess
    the chord of a supplied chord file.
    
    Returns a list:
    predicted chord, amplitude threshold used,
    set of chord notes found, strongest note
    """
    try:
        # Load audio file
        samples, sampling_rate = librosa.load(file_path, sr=None, mono=True)
        n = len(samples)

        # Compute FFT and normalise amplitude spectrum
        signal_fft = fft(samples)
        amplitude_spectrum = np.abs(signal_fft)
        amplitude_spectrum /= np.max(amplitude_spectrum)  # Normalise to [0, 1]

        # Get frequency bins (Hz)
        freqs = fftfreq(n, 1 / sampling_rate)
        
        # Set up variables for threshold finder
        auto_threshold = 1.0
        note_names_only = []
        root_guess = ''
        # Decrease threshold until 3 unique notes are found
        while len(set(note_names_only)) < 3 and auto_threshold > 0.0:
            # Decrement auto threshold
            auto_threshold -= 0.01
            
            # Thresholding to find dominant frequencies
            dominant_freq_indices = np.where(amplitude_spectrum[:n // 2] >= auto_threshold)[0]  # [:n // 2] positive freqs only
            dominant_freqs = freqs[dominant_freq_indices]

            # Convert frequencies to note names
            notes = [frequency_to_note_name(freq) for freq in dominant_freqs if freq > 0]
            note_names_only = [note[1] for note in notes]
            
            # If only one note found, store this as guessed root note
            if len(set(note_names_only)) == 1: root_guess = note_names_only[0]
        
        # Identify chord from detected notes
        chord_name = identify_chord(notes)

        # If identify_chord returns unknown or if too many freqs
        # got through threshold, set chord name as strongest note
        # as a best guess
        if chord_name == 'Unknown Chord' or chord_name == 'L':
            chord_name = root_guess

        return [chord_name, round(auto_threshold, 2), set(note_names_only), root_guess]

    except Exception as e:
        return f"Error during prediction: {e}"