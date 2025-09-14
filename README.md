# Piano Chord Transcriber

A web-based application that automatically transcribes chords from an audio file using either a signal-based FFT algorithm or a deep learning CNN model.

## Description

This project is a Flask-powered web application designed for musicians, producers, and hobbyists who want to quickly determine the chord progression of a piece of music. Users can upload an audio file, and the application will analyse it to produce a bar-by-bar chord chart. The system offers two distinct transcription methods: a fast, traditional Fast Fourier Transform (FFT) approach and a Convolutional Neural Network (CNN) model trained specifically on piano chords for higher accuracy on that instrument.

Once transcribed, the chords can be edited directly in the user-friendly interface and exported as either a simple text `.txt` file or a MusicXML `.xml` file, which can be imported into most music notation software.

## Features

* **Web-Based Interface:** Easy-to-use interface built with Flask.
* **Dual Transcription Models:**
    * **FFT Model:** A signal-processing algorithm that detects dominant frequencies to identify a wide range of chords (Major, Minor, Sus2, Sus4, Diminished, Augmented).
    * **CNN Model:** A TensorFlow/Keras model trained on a large dataset of piano chords for specialised transcription.
* **Automatic BPM Detection:** Automatically estimates the beats per minute (BPM) of the uploaded track to segment it correctly. Manual override is also available.
* **Interactive Chord Editor:** Double-click any chord in the results to edit it, allowing for easy correction and refinement.
* **Multiple Export Formats:**
    * **Plain Text (.txt):** For a simple, readable chord chart.
    * **MusicXML (.xml):** For importing into notation software like MuseScore or Sibelius.

## Technologies Used

* **Backend:** Python, Flask, Jinja2
* **Machine Learning:** TensorFlow, Keras, Scikit-learn
* **Audio Processing:** Librosa, NumPy, SoundFile
* **Frontend:** HTML, CSS (Pico.css), vanilla JavaScript

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing.

### Prerequisites

You will need Python 3.10.11 and pip installed on your system.

### Installation

1.  **Clone the repo**
    ```
    git clone https://gitlab.eeecs.qub.ac.uk/40463802/piano-chord-transcriber
    ```
2.  **Navigate to the project directory**
    ```
    cd your_project_name
    ```
3.  **(Recommended) Create and activate a virtual environment**
    ```
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```
4.  **Install the required packages**
    ```
    pip install -r requirements.txt
    ```
5.  **Place the Trained Model**
    Ensure the trained CNN model folder (e.g., `saved_crossval`) is present in the root directory of the project.
6.  **Run the application**
    ```
    flask run
    ```
    The application will be available at `http://127.0.0.1:5000`.

## Usage

1.  Open your web browser and navigate to `http://127.0.0.1:5000`.
2.  Click the "Choose File" button to upload a `.wav`, `.mp3`, or `.ogg` audio file.
3.  Select your desired transcription model (FFT or CNN).
4.  Choose between automatic BPM detection or manually entering a known BPM.
5.  Click the "Transcribe" button and wait for the process to complete.
6.  On the results page, you can play the audio, see the highlighted chords in real-time, and double-click any chord to make edits.
7.  Click "Save as Text" or "Save as XML" to download your chord chart.