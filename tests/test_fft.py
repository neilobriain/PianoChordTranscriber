"""
Unit tests for fft.py
"""

import os
import sys

# Add parent directory to the module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fft import get_chord_name

def test_get_chord_name_valid():
    """ Test that a chord prediction gets successfully returned. """
    chord = "tests/testchord.wav"
    actual = get_chord_name(chord)[0]
    expected = "Gmaj"
    assert actual == expected
    
def test_predict_chord_invalid():
    """ Tests that function returns an error message on failure. """
    chord = None
    actual = get_chord_name(chord)[:5]
    expected = "Error"
    assert actual == expected