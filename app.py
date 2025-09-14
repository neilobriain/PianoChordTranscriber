import os
from flask import Flask, request, render_template, send_from_directory
from utils import get_chord_list, auto_predict_bpm, estimate_key
from error_handling import is_compatible_audio

UPLOAD_FOLDER = 'uploads'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Render index page in browser
@app.route('/')
def index():
    return render_template('index.html')

# Render instructions page in browser
@app.route('/instructions')
def instructions():
    return render_template('instructions.html')

# Audio upload, transcribe, render results on transcribe page
@app.route('/', methods=['POST'])
def file_upload():
    file = request.files['audio'] # gets audio file uploaded from index.html form
    auto_bpm_check = request.form.get('auto_bpm')
    bpm_selection = request.form.get('bpm_selection', '120') 
    # bpm = int(bpm_selection)
    model_selection = request.form.get('model', 'fft')
    print(model_selection)
    
    if file:
        # Check is a compatible file
        if not is_compatible_audio(file.filename):
            return render_template('error.html', error_heading='Invalid Upload', error_message='Accepted file types: WAV, MP3, and Ogg Vorbis.')
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Set BPM (auto or manual)
        if auto_bpm_check == 'yes':
            auto_bpm = auto_predict_bpm(filepath)
            if auto_bpm[0] == 'success':
                bpm = auto_bpm[1]
            else: bpm = int(bpm_selection) # Set to manual if auto but fails
        else: bpm = int(bpm_selection) # Set to manual selection if not auto

        chords = get_chord_list(filepath, bpm, model_selection)
        print(chords[1])
        estimated_key = estimate_key(chords[1])
        
        indexed_chords = list(enumerate(chords[1])) # index for cell id to highlight in html
        
        print(file.filename)
        return render_template('transcribe.html', indexed_chords=indexed_chords, filename=file.filename, bpm=bpm, model=model_selection, estimated_key=estimated_key)

# Needed for sending uploaded audio file to audio player 
@app.route('/uploads/<path:filename>')
def uploads(filename):
    return send_from_directory('uploads', filename)

# 404 error page
@app.errorhandler(404)
def page_not_found(error):
  return render_template('error.html', error_heading='404', error_message='The requested page could not be found.')

if __name__ == '__main__':
    app.run(debug=True)