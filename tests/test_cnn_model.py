"""
Unit tests for cnn_model.py
"""

import os
import sys

# Add parent directory to the module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cnn_model import predict_chord

def test_predict_chord_valid():
    """ Test that a chord prediction gets successfully returned. """
    chord = "tests/testchord.wav"
    actual = predict_chord(chord)[0]
    expected = "Gmaj"
    assert actual == expected

def test_predict_chord_invalid():
    """ Tests that function returns an error message on failure. """
    chord = None
    actual = predict_chord(chord)[:5]
    expected = "Error"
    assert actual == expected