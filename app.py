# app.py
import os
import pickle
import librosa
import pandas as pd
import numpy as np
from keras import models
from keras import layers
import keras
from keras.models import Model
from flask import Flask, render_template, request

app = Flask(__name__)
global up_file
# Set the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def audio_preprocessing(file):
    #header = 'filename chroma_stft rmse spec_cent spec_bw rolloff zcr mfcc lpc label'
    dict_data = {}
            
    #dict_data['filename'] = file.filename

    y, sr = librosa.load(file, mono=True)
    y, index = librosa.effects.trim(y)

    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_stft = chroma_stft.mean(axis=1)
    for i in range(1, chroma_stft.shape[0]+1):
        dict_data['chroma_stft'+str(i)] = chroma_stft[i-1]

    rmse = librosa.feature.rms(y=y)
    dict_data['rmse'] = rmse.mean()

    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    dict_data['spec_cent'] = spec_cent.mean()

    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    dict_data['spec_bw'] = spec_bw.mean()

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    dict_data['rolloff'] = rolloff.mean()

    zcr = librosa.feature.zero_crossing_rate(y)
    dict_data['zcr'] = zcr.mean()

    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    mfcc = mfcc.mean(axis=1)
    for i in range(1, mfcc.shape[0]+1):
        dict_data['mfcc'+str(i)] = mfcc[i-1]

    lpc = librosa.lpc(y=y, order=2)
    #for i in range(1, lpc.shape[0]):
    #    dict_data['lpc'+str(i)].append(lpc[i])
    dict_data['lpc'] = lpc[1]
    
    df = pd.DataFrame(dict_data, index=[0])

    with open('scaler.pkl','rb') as f:
        sc = pickle.load(f)
    df = sc.transform(df.values)
    
    print('Preproessing done..')
    return df
    
# Define a route to serve the HTML page with the form
@app.route('/')
def index():
    return render_template('index.html')


# Define the route to handle file upload
@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the POST request has the file part
    if 'audio_file' not in request.files:
        return 'No file part'
    
    up_file = request.files['audio_file']
    
    # If the user does not select a file, the browser submits an empty file without a filename
    if up_file.filename == '':
        return 'No selected file'
    
    # If the file exists and is an allowed format
    if up_file and allowed_file(up_file.filename):
        # Save the file to the upload folder
        filename = up_file.filename
        up_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        file_loc = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        prep_data = audio_preprocessing(file_loc)
        saved_model = keras.models.load_model('model.h5')
        score = saved_model.predict(prep_data.reshape(-1,38))
        output = np.argmax(score)
        return render_template('index.html', input_text='File '+filename+' uploaded successfully.', output_text='Predicted Speakder: '+'Speaker_'+str(output))
    
    return 'Invalid file format'

# Function to check if the file format is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'wav', 'flac'}

if __name__ == '__main__':
    app.run(debug=True)

    
