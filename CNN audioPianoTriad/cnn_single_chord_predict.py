"""
Utility script to check a single example of a Gmin chord within the
dataset against the newly trained CNN model, to ensure the model can be
loaded and run.
"""

import tensorflow as tf
import pathlib

DATASET_PATH = r'D:\datasets\audioPianoTriadDataset\audioPianoTriadDataset\Chords\MajMin'
data_dir = pathlib.Path(DATASET_PATH)

imported = tf.saved_model.load("saved")
print(imported(tf.constant(str(data_dir/'Gmin/piano_3_Gn_n_f_06.wav'))))