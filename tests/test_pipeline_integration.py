"""
Integration tests for the transcription pipeline.
"""

import os
import sys

# Add parent directory to the module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import get_chord_list

def test_transcription_pipeline_valid():
    """
    Tests the full pipeline from audio input to chord list output.
    This verifies that get_chord_list correctly integrates the audio splitting
    and chord prediction modules (e.g. fft.py) to produce a valid output.
    """
    # Arrange: Define inputs for the pipeline
    song_path = "tests/testsong.wav"
    bpm = 120
    model = 'fft'

    # Act: Run the full transcription process
    status, chords = get_chord_list(song_path, bpm, model)

    # Assert: Check the results of the integration
    assert status == "success"
    assert isinstance(chords, list)
    assert len(chords) > 0

    # Verify that the inner elements are also lists, as expected from the prediction functions
    assert isinstance(chords[0], list)

def test_transcription_pipeline_invalid():
    """
    Tests that the pipeline fails gracefully when an exception is encountered.
    """
    song_path = None
    bpm = 120
    model = 'fft'

    status, message = get_chord_list(song_path, bpm, model)

    # Assert that the pipeline gracefully fails and returns the status and an error message
    assert status == "fail"
    assert isinstance(message, str)