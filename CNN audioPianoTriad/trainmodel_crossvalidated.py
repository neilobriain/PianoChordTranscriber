"""
Training script for CNN model on audioPianoTriad (Augmented x10) Major/Minor
chords dataset, encompassing k-fold cross-validation. This is the final model
used in the project. Model is exported to the 'saved_crossval' folder.
"""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
from sklearn.model_selection import KFold

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Cross-validation parameters
K_FOLDS = 5
BATCH_SIZE = 32  # Reduced from 64 to save memory

# Use your local dataset directory
DATASET_PATH = r'D:\datasets\audioPianoTriadDataset\audioPianoTriadDataset\Chords\MajMin'
data_dir = pathlib.Path(DATASET_PATH)

# List subfolders (chord class names)
commands = np.array([item.name for item in data_dir.iterdir() if item.is_dir()])
print('Commands:', commands)

# Get all file paths and labels for cross-validation
def get_all_file_paths_and_labels():
    file_paths = []
    labels = []
    class_names = sorted([item.name for item in data_dir.iterdir() if item.is_dir()])
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = data_dir / class_name
        for file_path in class_dir.glob('*.wav'):
            file_paths.append(str(file_path))
            labels.append(class_idx)
    
    return np.array(file_paths), np.array(labels), class_names

print("Getting file paths...")
all_file_paths, all_labels, label_names = get_all_file_paths_and_labels()
label_names = np.array(label_names)
print(f"Found {len(all_file_paths)} files")
print("label names:", label_names)
np.save('label_names.npy', label_names)

'''
Create a utility function for converting waveforms to spectrograms:
'''
def get_spectrogram(waveform):
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

def load_and_preprocess_audio(file_path, label):
    # Load audio file
    audio = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(audio, desired_channels=1, desired_samples=64000)
    audio = tf.squeeze(audio, axis=-1)
    
    # Convert to spectrogram
    spectrogram = get_spectrogram(audio)
    return spectrogram, label

def create_dataset_from_paths(file_paths, labels, batch_size=32):
    """Create a TensorFlow dataset from file paths and labels"""
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.map(load_and_preprocess_audio, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

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

def create_model(input_shape, num_labels):
    # Instantiate the `tf.keras.layers.Normalization` layer.
    norm_layer = layers.Normalization()
    
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
    
    '''
    Configure the Keras model with the Adam optimizer and the cross-entropy loss:
    '''
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )
    
    return model, norm_layer

# Cross-validation setup
kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=seed)
cv_scores = []
cv_histories = []

# Get input shape from a sample
print("Getting input shape...")
sample_dataset = create_dataset_from_paths(all_file_paths[:BATCH_SIZE], all_labels[:BATCH_SIZE], BATCH_SIZE)
for example_spectrograms, example_spect_labels in sample_dataset.take(1):
    input_shape = example_spectrograms.shape[1:]
    break
print('Input shape:', input_shape)
num_labels = len(label_names)

# Perform cross-validation
for fold, (train_idx, val_idx) in enumerate(kfold.split(all_file_paths)):
    print(f"\n=== FOLD {fold + 1}/{K_FOLDS} ===")
    print(f"Training samples: {len(train_idx)}, Validation samples: {len(val_idx)}")
    
    # Split file paths for this fold
    train_paths, train_labels_fold = all_file_paths[train_idx], all_labels[train_idx]
    val_paths, val_labels_fold = all_file_paths[val_idx], all_labels[val_idx]
    
    # Create datasets from file paths (more memory efficient)
    print("Creating training dataset...")
    train_ds = create_dataset_from_paths(train_paths, train_labels_fold, BATCH_SIZE)
    train_ds = train_ds.shuffle(1000)  # Shuffle the dataset
    
    print("Creating validation dataset...")
    val_ds = create_dataset_from_paths(val_paths, val_labels_fold, BATCH_SIZE)
    
    # Create and compile model
    print("Creating model...")
    model, norm_layer = create_model(input_shape, num_labels)
    
    # Fit the normalization layer to the training data
    print("Adapting normalization layer...")
    norm_layer.adapt(data=train_ds.map(lambda spec, label: spec))
    
    if fold == 0:  # Only show summary for first fold
        model.summary()
    
    '''
    Train the model over 10 epochs:
    '''
    print("Starting training...")
    EPOCHS = 10
    try:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
            verbose=1 if fold == 0 else 2  # Less verbose for subsequent folds
        )
        
        # Evaluate the model on validation data
        val_loss, val_accuracy = model.evaluate(val_ds, verbose=0)
        cv_scores.append(val_accuracy)
        cv_histories.append(history)
        
        print(f"Fold {fold + 1} - Validation Accuracy: {val_accuracy:.4f}")
        
        # Save model for this fold
        model.save(f'newtriad1majmin_1907_fold_{fold + 1}.keras')
        
    except Exception as e:
        print(f"Error during training fold {fold + 1}: {e}")
        cv_scores.append(0.0)  # Add placeholder score
        continue

# Print cross-validation results
print(f"\n=== CROSS-VALIDATION RESULTS ===")
valid_scores = [score for score in cv_scores if score > 0]
if valid_scores:
    print(f"Fold accuracies: {[f'{score:.4f}' for score in cv_scores]}")
    print(f"Mean CV accuracy: {np.mean(valid_scores):.4f} (+/- {np.std(valid_scores) * 2:.4f})")
    print(f"Best fold accuracy: {np.max(valid_scores):.4f}")
else:
    print("No successful folds completed.")

# Train final model on all data
print(f"\n=== TRAINING FINAL MODEL ON SUBSET OF DATA ===")

# Use a smaller subset for final training to avoid memory issues
subset_size = min(10000, len(all_file_paths))  # Use max 10k samples for final model
indices = np.random.choice(len(all_file_paths), size=subset_size, replace=False)
subset_paths = all_file_paths[indices]
subset_labels = all_labels[indices]

# Create a test set from 20% of subset
test_size = int(0.2 * len(subset_paths))
test_paths, test_labels = subset_paths[-test_size:], subset_labels[-test_size:]
train_paths_final, train_labels_final = subset_paths[:-test_size], subset_labels[:-test_size]

print(f"Final training set: {len(train_paths_final)} samples")
print(f"Test set: {len(test_paths)} samples")

train_ds_final = create_dataset_from_paths(train_paths_final, train_labels_final, BATCH_SIZE)
test_ds = create_dataset_from_paths(test_paths, test_labels, BATCH_SIZE)

train_ds_final = train_ds_final.shuffle(1000)

# Create final model
print("Creating final model...")
model, norm_layer = create_model(input_shape, num_labels)

# Fit the state of the layer to the spectrograms with `Normalization.adapt`.
print("Adapting normalization layer for final model...")
norm_layer.adapt(data=train_ds_final.map(lambda spec, label: spec))

model.summary()

'''
Train the final model:
'''
print("Training final model...")
EPOCHS = 10
history = model.fit(
    train_ds_final,
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
)

'''
Plot the training loss curves for the final model:
'''
metrics = history.history
plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
plt.plot(history.epoch, metrics['loss'])
plt.legend(['loss'])
plt.ylim([0, max(plt.ylim())])
plt.xlabel('Epoch')
plt.ylabel('Loss [CrossEntropy]')

plt.subplot(1,2,2)
plt.plot(history.epoch, 100*np.array(metrics['accuracy']))
plt.legend(['accuracy'])
plt.ylim([0, 100])
plt.xlabel('Epoch')
plt.ylabel('Accuracy [%]')
plt.show()

'''
Evaluate the model performance on test set
'''
print("Evaluating final model...")
final_results = model.evaluate(test_ds, return_dict=True)
print(f"Final model test accuracy: {final_results['accuracy']:.4f}")

'''
Display a confusion matrix
'''
print("Generating confusion matrix...")
y_pred = model.predict(test_ds)
y_pred = tf.argmax(y_pred, axis=1)
y_true = tf.concat(list(test_ds.map(lambda s,lab: lab)), axis=0)
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
print("Testing export model...")
export = ExportModel(model)
test_file = data_dir/'Dmaj'
test_files = list(test_file.glob('*.wav'))
if test_files:
    print(export(tf.constant(str(test_files[0]))))

'''
Save and reload the model, the reloaded model gives identical output:
'''
print("Saving export model...")
tf.saved_model.save(export, "saved_crossval")
imported = tf.saved_model.load("saved_crossval")
if test_files:
    print(imported(tf.constant(str(test_files[0]))))