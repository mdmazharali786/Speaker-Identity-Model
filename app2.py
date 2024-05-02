# app.py
import os
import pickle
import numpy as np
import keras
import torch
from sklearn.preprocessing import LabelEncoder
from keras import layers
from keras.models import Model
from pyannote.audio import Pipeline, Audio
from flask import Flask, render_template, request
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding


app = Flask(__name__)
global up_file

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

embedding_model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb",
    device=torch.device("cuda"))

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", 
                                    use_auth_token='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')

def audio_preprocessing(file):
    audio = Audio()
    waveform, sample_rate = audio(file)
    if waveform.shape[0]!=1:
        waveform = waveform[:1,:]
    dia = pipeline.to(torch.device(0))({"waveform": waveform, "sample_rate": sample_rate}, num_speakers=3)
    embedded_audio = []
    for speech_turn, track, speaker in dia.itertracks(yield_label=True):
        if speech_turn.end-speech_turn.start>=1:
            start = round(speech_turn.start*sample_rate)
            end = round(speech_turn.end*sample_rate)
            data = waveform[:,start:end]
            data = embedding_model(data[None])
            data = np.nan_to_num(data)
            data = data.flatten()
            embedded_audio.append(data)
    
    embedded_audio = np.array(embedded_audio)
    
    with open('model/scaler.pkl', 'rb') as sc:
        std_scaler = pickle.load(sc)
    embedded_audio = std_scaler.transform(embedded_audio)
    return embedded_audio
    
def predict_speaker(emb_data):
    saved_model = keras.models.load_model('model/model.h5')
    with open('model/label_enc.pkl', 'rb') as le:
        label_encoder = pickle.load(le)
    speaker_set_with_score = {0:[], 1:[], 2:[], 3:[], 4:[]}
    scores = saved_model.predict(emb_data)
    for i in range(scores.shape[0]):
        idx = np.argmax(scores[i])
        pred_score = scores[i][idx]
        speaker_set_with_score[idx].append(pred_score)
    #print(speaker_set_with_score)
    speaker_set = set()
    for spkr, proba in speaker_set_with_score.items():
        if np.median(proba)>=0.5:
            speaker = label_encoder.inverse_transform([spkr])[0]
        else:
            speaker = 'Unknown'
        speaker_set.add(speaker)
    return speaker_set

@app.route('/')
def index():
    return render_template('index.html')


    
@app.route('/upload', methods=['POST'])
def upload_file():

    if 'audio_file' not in request.files:
        return 'No file part'
    
    up_file = request.files['audio_file']
    

    if up_file.filename == '':
        return 'No selected file'
    

    if up_file and allowed_file(up_file.filename):

        filename = up_file.filename
        up_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        file_loc = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        embedded_data = audio_preprocessing(file_loc)
        speaker_list = predict_speaker(embedded_data)
        speaker_list = list(speaker_list)
        #print(speaker_list)
        return render_template('index.html', pred_speaker=speaker_list)
    
    return 'Invalid file format'


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'wav', 'flac', 'mp3', 'mp4', 'm4a'}

if __name__ == '__main__':
    app.run()