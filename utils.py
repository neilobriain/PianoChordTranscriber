import os
import math
import librosa
import soundfile as sf
from collections import Counter
from fft import get_chord_name
from cnn_model import predict_chord

def auto_predict_bpm(audio_path):
    """
    Automatically predicts the beats per minute (BPM)
    for a supplied audio file.
    
    Returns a list:
    [0] - success or fail
    [1] - estimated BPM as int, or error message    
    """
    try:
        # Load the audio file
        y, sr = librosa.load(audio_path)

        # Estimate tempo and beat frames
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        tempo = round(tempo)
        
        if tempo == 0: raise Exception("Zero tempo found")
        
        # Try to keep tempo in usable range for transcription
        if tempo < 40:
            tempo *= 2
        elif tempo > 240:
            tempo /= 2
        
        return ["success", tempo]

    except Exception as e:
        print(f"Failure - auto_predict_bpm: {e}")
        return ["fail", f"Error in auto_predict_bpm function: {e}"]

def split_audio_by_bars(audio_path, bpm):
    """
    Splits an audio file into separate files for each beat and saves
    them in the uploads/split folder.
    
    These can then be loaded for processing in the transcriber.
    
    Returns a list:
    [0] - success or fail
    [1] - total no. of bars or error message
    """
    try:
        output_dir="uploads/split"
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)

        # Calculate bar duration
        seconds_per_beat = 60 / bpm
        seconds_per_bar = seconds_per_beat * 4  # 4/4 time
        samples_per_bar = int(sr * seconds_per_bar)

        # Total bars
        total_bars = math.ceil(len(y) / samples_per_bar)

        # Make output directory
        os.makedirs(output_dir, exist_ok=True)

        # Split and save
        for i in range(total_bars):
            start_sample = i * samples_per_bar
            end_sample = min((i + 1) * samples_per_bar, len(y))
            bar_audio = y[start_sample:end_sample]

            filename = os.path.join(output_dir, f"bar_{i+1:03d}.wav")
            sf.write(filename, bar_audio, sr)

        print(f"Success - split_audio_by_bars. Total bars: {total_bars}")
        return ["success", total_bars]
    
    except Exception as e:
        print(f"Failure - split_audio_by_bars: {e}")
        return ["fail", f"Error in split_audio_by_bars function: {e}"]

def get_chord_list(audio_path, bpm, model_selection):
    """
    Returns a list of predicted chords and associated information from an audio file.
     
    Returns a list:
    [0] - success or fail
    [1] - list of predicted chords or error message  
    """
    chord_list = []
    chord = ''
    try:
        bars = split_audio_by_bars(audio_path, bpm)
        
        # Raise exception if audio could not be split into bars
        if not bars[0] == "success":
            raise Exception(f"get_chord_list could not get successful outcome from split_audio_by_bars: {bars[1]}")
        
        # Iterate through total no. of bars, transcribe, and append to list
        # if CNN model selected, use that, otherwise default to FFT
        if model_selection=='cnn':
            for i in range(bars[1]):
                chord = predict_chord(f"uploads/split/bar_{i+1:03d}.wav")
                chord_list.append(chord)
        else:
            for i in range(bars[1]):
                chord = get_chord_name(f"uploads/split/bar_{i+1:03d}.wav")
                chord_list.append(chord)
        
        return ["success", chord_list]
    
    except Exception as e:
        return ["fail", f"Error in get_chord_list function: {e}"]

def estimate_key(chords_list):
    """
    Returns the estimated music key centre by finding the most featured element in the chords list.
    """
    try:
        chords = [chord[0] for chord in chords_list] # extract just the chord names
        counter = Counter(chords)
        return counter.most_common(1)[0][0] # just the chord name of most common
        
    except Exception as e:
        print(f"Key estimation failure - {e}")
        return ''