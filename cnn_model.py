import tensorflow as tf
import numpy as np

def predict_chord(filename):
    """
    Uses the CNN model to guess
    the chord of a supplied chord file.
    
    Returns a list:
    predicted chord, confidence in prediction
    """
    try:
        imported = tf.saved_model.load("saved_crossval")
        result = imported(tf.constant(filename))
        
        # Access predicted class name (chord)
        chord_name = result['class_names'].numpy()[0].decode()
        
        logits = result['predictions'].numpy()[0]
        probabilities = tf.nn.softmax(logits).numpy()
        confidence = np.max(probabilities)
        
        return [chord_name, f"{confidence:.2%}"]

    except Exception as e:
        return f"Error during prediction: {e}"