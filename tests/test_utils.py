"""
Unit tests for utils.py
"""

import os
import sys

# Add parent directory to the module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import estimate_key, get_chord_list, split_audio_by_bars, auto_predict_bpm

def test_estimate_key_valid():
    """ Tests that estimated key is returned based on chords list. """
    chords = ["G", "G", "C"]
    actual = estimate_key(chords)
    expected = "G"
    assert actual == expected

def test_estimate_key_invalid():
    """ Tests that function fails graciously if it cannot return an estimate. """
    chords = None
    actual = estimate_key(chords)
    expected = ""
    assert actual == expected
    
def test_get_chord_list_valid():
    """ Tests that a chord list gets successfully returned. """
    song = "tests/testsong.wav"
    actual = get_chord_list(song, 120, 'fft')[0]
    expected = "success"
    assert actual == expected

def test_get_chord_list_invalid():
    """ Tests that a failure note gets successfully returned if invalid. """
    song = None
    actual = get_chord_list(song, 120, 'fft')[0]
    expected = "fail"
    assert actual == expected
    
def test_split_audio_by_bars_valid():
    """ Tests that audio gets successfully split into bars. """
    song = "tests/testsong.wav"
    actual = split_audio_by_bars(song, 120)[0]
    expected = "success"
    assert actual == expected

def test_split_audio_by_bars_invalid():
    """ Tests that a failure note gets successfully returned if invalid. """
    song = None
    actual = split_audio_by_bars(song, 120)[0]
    expected = "fail"
    assert actual == expected
    
def test_auto_predict_bpm_valid():
    """ Tests that bpm can be auto predicted. """
    song = "tests/testsong.wav"
    actual = auto_predict_bpm(song)[1]
    expected = 60
    assert actual == expected
    
def test_auto_predict_bpm_invalid():
    """ Tests that a failure note gets successfully returned if invalid. """
    song = None
    actual = auto_predict_bpm(song)[0]
    expected = "fail"
    assert actual == expected