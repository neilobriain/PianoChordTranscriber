"""
Unit tests for error_handling.py
"""

import os
import sys

# Add parent directory to the module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from error_handling import is_compatible_audio

def test_is_compatible_audio_valid():
    """ Tests that valid file extensions are recognised as such. """
    filenames_valid = ["test.mp3", "test.ogg", "test.wav"]
    for filename in filenames_valid:
        assert is_compatible_audio(filename)

def test_is_compatible_audio_invalid():
    """ Tests that invalid file extensions are recognised as such. """
    filenames_invalid = ["test.mp4", "test.txt", "test.mov"]
    for filename in filenames_invalid:
        assert not is_compatible_audio(filename)