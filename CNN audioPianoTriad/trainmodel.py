"""
Training script for CNN model on audioPianoTriad (Augmented x10) Major/Minor
chords dataset. This model is not cross validated and so is not the final
model used in the project. Model is exported to the 'saved' folder.
"""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)


# Use your local dataset directory
DATASET_PATH = r'D:\datasets\audioPianoTriadDataset\audioPianoTriadDataset\Chords\MajMin'
data_dir = pathlib.Path(DATASET_PATH)

# List subfolders (chord class names)
commands = np.array([item.name for item in data_dir.iterdir() if item.is_dir()])
print('Commands:', commands)


"""
Divided into directories this way, you can easily load the data using
keras.utils.audio_dataset_from_directory.
"""
train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=data_dir,
    batch_size=64,
    validation_split=0.2,
    seed=0,
    output_sequence_length=64000, # adapted to 4 seconds for my dataset
    subset='both')

label_names = np.array(train_ds.class_names)
print()
print("label names:", label_names)
np.save('label_names.npy', label_names)

'''
The dataset now contains batches of audio clips and integer labels.
The audio clips have a shape of (batch, samples, channels). 
'''
print(train_ds.element_spec)

'''
This dataset only contains single channel audio, so use the tf.squeeze function to drop the extra axis:
'''
def squeeze(audio, labels):
  audio = tf.squeeze(audio, axis=-1)
  return audio, labels

train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

'''
The utils.audio_dataset_from_directory function only returns up to two splits.
It's a good idea to keep a test set separate from your validation set.
Dataset.shard is used to split the validation set into two halves.
Note that iterating over any shard will load all the data, and only keep its fraction. 
'''
test_ds = val_ds.shard(num_shards=2, index=0)
val_ds = val_ds.shard(num_shards=2, index=1)

for example_audio, example_labels in train_ds.take(1):  
  print(example_audio.shape)
  print(example_labels.shape)

'''
Plot a few waveforms to check
'''
print(label_names[[1,1,3,0]])

plt.figure(figsize=(16, 10))
rows = 3
cols = 3
n = rows * cols
for i in range(n):
  plt.subplot(rows, cols, i+1)
  audio_signal = example_audio[i]
  plt.plot(audio_signal)
  plt.title(label_names[example_labels[i]])
  plt.yticks(np.arange(-1.2, 1.2, 0.2))
  plt.ylim([-1.1, 1.1])
plt.show()

def get_spectrogram(waveform):
  '''
  A utility function for converting waveforms to spectrograms:

  The waveforms need to be of the same length, so that when you convert them to spectrograms,
  the results have similar dimensions. This is done by zero-padding the audio clips
  that are shorter than one second (using tf.zeros).
  
  When calling tf.signal.stft, choose the frame_length and frame_step parameters such that the
  generated spectrogram "image" is almost square.
  
  The STFT produces an array of complex numbers representing magnitude and phase.
  In this case we only need the magnitude, which we derive by applying
  tf.abs on the output of tf.signal.stft.
  '''
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(
      waveform, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram


'''
Next, start exploring the data. Print the shapes of one example's tensorized waveform
and the corresponding spectrogram.
'''
for i in range(3):
  label = label_names[example_labels[i]]
  waveform = example_audio[i]
  spectrogram = get_spectrogram(waveform)

  print('Label:', label)
  print('Waveform shape:', waveform.shape)
  print('Spectrogram shape:', spectrogram.shape)
  print('Audio playback')

'''
Now, define a function for displaying a spectrogram:
'''
def plot_spectrogram(spectrogram, ax):
  if len(spectrogram.shape) > 2:
    assert len(spectrogram.shape) == 3
    spectrogram = np.squeeze(spectrogram, axis=-1)
  # Convert the frequencies to log scale and transpose, so that the time is
  # represented on the x-axis (columns).
  # Add an epsilon to avoid taking a log of zero.
  log_spec = np.log(spectrogram.T + np.finfo(float).eps)
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)

'''
Plot the example's waveform over time and the corresponding spectrogram (frequencies over time):
'''
fig, axes = plt.subplots(2, figsize=(12, 8))
timescale = np.arange(waveform.shape[0])
axes[0].plot(timescale, waveform.numpy())
axes[0].set_title('Waveform')
axes[0].set_xlim([0, 16000])

plot_spectrogram(spectrogram.numpy(), axes[1])
axes[1].set_title('Spectrogram')
plt.suptitle(label.title())
plt.show()

'''
Now, create spectrogram datasets from the audio datasets
'''
def make_spec_ds(ds):
  return ds.map(
      map_func=lambda audio,label: (get_spectrogram(audio), label),
      num_parallel_calls=tf.data.AUTOTUNE)

train_spectrogram_ds = make_spec_ds(train_ds)
val_spectrogram_ds = make_spec_ds(val_ds)
test_spectrogram_ds = make_spec_ds(test_ds)

'''
Examine the spectrograms for different examples of the dataset:
'''
for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
  break

rows = 3
cols = 3
n = rows*cols
fig, axes = plt.subplots(rows, cols, figsize=(16, 9))

for i in range(n):
    r = i // cols
    c = i % cols
    ax = axes[r][c]
    plot_spectrogram(example_spectrograms[i].numpy(), ax)
    ax.set_title(label_names[example_spect_labels[i].numpy()])

plt.show()

# Build and train the model
'''
Add Dataset.cache and Dataset.prefetch operations to reduce read latency while training the model:
'''
train_spectrogram_ds = train_spectrogram_ds.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
val_spectrogram_ds = val_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
test_spectrogram_ds = test_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)

'''
For the model, a convolutional neural network (CNN) is used as the audio files
have been transformed into spectrogram images.

The tf.keras.Sequential model will use the following Keras preprocessing layers:

    tf.keras.layers.Resizing: to downsample the input to enable the model to train faster.
    tf.keras.layers.Normalization: to normalize each pixel in the image based on its mean and standard deviation.

For the Normalization layer, its adapt method would first need to be called on the training data
in order to compute aggregate statistics (that is, the mean and the standard deviation).
'''

input_shape = example_spectrograms.shape[1:]
print('Input shape:', input_shape)
num_labels = len(label_names)

# Instantiate the `tf.keras.layers.Normalization` layer.
norm_layer = layers.Normalization()
# Fit the state of the layer to the spectrograms
# with `Normalization.adapt`.
norm_layer.adapt(data=train_spectrogram_ds.map(map_func=lambda spec, label: spec))

model = models.Sequential([
    layers.Input(shape=input_shape),
    # Downsample the input
    layers.Resizing(32, 32),
    # Normalise
    norm_layer,
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_labels),
])

model.summary()

'''
Configure the Keras model with the Adam optimizer and the cross-entropy loss:
'''
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)
'''
Train the model over 10 epochs:
'''
EPOCHS = 10
history = model.fit(
    train_spectrogram_ds,
    validation_data=val_spectrogram_ds,
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
)

'''
Plot the training and validation loss curves to
check how the model has improved during training:
'''
metrics = history.history
plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.ylim([0, max(plt.ylim())])
plt.xlabel('Epoch')
plt.ylabel('Loss [CrossEntropy]')

plt.subplot(1,2,2)
plt.plot(history.epoch, 100*np.array(metrics['accuracy']), 100*np.array(metrics['val_accuracy']))
plt.legend(['accuracy', 'val_accuracy'])
plt.ylim([0, 100])
plt.xlabel('Epoch')
plt.ylabel('Accuracy [%]')
plt.show()

'''
Evaluate the model performance

Run the model on the test set and check the model's performance:
'''
model.evaluate(test_spectrogram_ds, return_dict=True)

'''
Display a confusion matrix
'''
y_pred = model.predict(test_spectrogram_ds)
y_pred = tf.argmax(y_pred, axis=1)
y_true = tf.concat(list(test_spectrogram_ds.map(lambda s,lab: lab)), axis=0)
confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx,
            xticklabels=label_names,
            yticklabels=label_names,
            annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()

'''
Export the model with preprocessing
'''
class ExportModel(tf.Module):
  def __init__(self, model):
    self.model = model

    # Accept either a string-filename or a batch of waveforms.
    self.__call__.get_concrete_function(
        x=tf.TensorSpec(shape=(), dtype=tf.string))
    self.__call__.get_concrete_function(
       x=tf.TensorSpec(shape=[None, 64000], dtype=tf.float32))


  @tf.function
  def __call__(self, x):
    # If they pass a string, load the file and decode it. 
    if x.dtype == tf.string:
      x = tf.io.read_file(x)
      x, _ = tf.audio.decode_wav(x, desired_channels=1, desired_samples=64000,)
      x = tf.squeeze(x, axis=-1)
      x = x[tf.newaxis, :]

    x = get_spectrogram(x)  
    result = self.model(x, training=False)

    class_ids = tf.argmax(result, axis=-1)
    class_names = tf.gather(label_names, class_ids)
    return {'predictions':result,
            'class_ids': class_ids,
            'class_names': class_names}

'''
Test run the "export" model
'''
export = ExportModel(model)
print(export(tf.constant(str(data_dir/'Dmaj/piano_3_Dn_j_f_00.wav'))))

'''
Save and reload the model, the reloaded model gives identical output:
'''
tf.saved_model.save(export, "saved")
imported = tf.saved_model.load("saved")
print(imported(tf.constant(str(data_dir/'Dmaj/piano_3_Dn_j_f_00.wav'))))