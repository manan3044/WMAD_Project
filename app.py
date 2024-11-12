from flask import Flask, request, jsonify, send_file, abort
from flask_cors import CORS
from pydub import AudioSegment
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import os
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import io

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests for React Native

# Load your pre-trained model
model = load_model('My_Best_Model.h5')
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

model_2 = load_model('fast_text_model.h5')
model_2.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Load the tokenizers and model (assuming they're saved as shown earlier)
with open('ft_word_tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('ft_char_tokenizer.pkl', 'rb') as handle:
    char_tokenizer = pickle.load(handle)

# Define the segment duration (500 milliseconds)
segment_duration = 500

# Azure Blob Storage connection
connection_string = "DefaultEndpointsProtocol=https;AccountName=wmadprojectcontainer;AccountKey=AVZbuCTn7xKm4Ere7oGSfvC3JTxuBca9j76yripqxok3J9xKiYiIomY6sX3qHFP0O5oTjoNkTJia+AStFGlfmg==;EndpointSuffix=core.windows.net"
container_name = "audiofiles"
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client(container_name)

# Function to extract features from audio segments
def extract_features_from_segment(segment):
    try:
        samples = np.array(segment.get_array_of_samples()).astype(np.float32)
        mfccs = librosa.feature.mfcc(y=samples, sr=segment.frame_rate, n_mfcc=15)
        mfccs = np.mean(mfccs.T, axis=0)
        return mfccs
    except Exception as e:
        print(f"Error processing segment: {e}")
        return None

# Function to process the audio file and mute bad words
def process_audio(audio):
    global audio_file_name
    try:
        bad_word_indices = []  # Store indices of bad word segments
        segments = []  # To store MFCC features for each segment

        # Split the audio into segments and process each segment
        for i in range(0, len(audio), segment_duration):
            segment = audio[i:i + segment_duration]
            mfccs = extract_features_from_segment(segment)
            if mfccs is not None:
                segments.append(mfccs)

        # Prepare the input for the model
        X_test = np.array(segments)

        if len(X_test) > 0:
            # Get predictions from the model
            predictions = model.predict(X_test)
            
            # Check for bad word predictions (Assuming 1 means bad word)
            for i, pred in enumerate(predictions):
                if np.argmax(pred) == 1:  # Assuming bad word is labeled as 1
                    start_time = i * segment_duration
                    end_time = start_time + segment_duration
                    bad_word_indices.append((start_time, end_time))

            # Mute the bad words in the audio
            for start, end in bad_word_indices:
                audio = audio[:start] + AudioSegment.silent(duration=end - start) + audio[end:]

            # Save the processed (muted) audio to Azure Blob Storage
            processed_audio_path = f"muted_audio_{audio_file_name}"
            processed_audio_buffer = io.BytesIO()
            audio.export(processed_audio_buffer, format="wav")
            processed_audio_buffer.seek(0)

            blob_client = container_client.get_blob_client(processed_audio_path)
            blob_client.upload_blob(processed_audio_buffer, overwrite=True)

            return processed_audio_path
        else:
            print("No valid segments to predict.")
            return None
    except Exception as e:
        print(f"Error processing audio file: {e}")
        return None

#------------------------------------------Text------------------------------------------
def preprocess_sentence(sentence):
    char_max_length = 15
    max_length = 475
    word_sequence = tokenizer.texts_to_sequences([sentence])
    padded_word_sequence = pad_sequences(word_sequence, maxlen=max_length, padding='post', truncating='post')
    char_sequence = [[char_tokenizer.word_index.get(char, 0) for char in word] for word in sentence.split()]
    char_sequence = pad_sequences(char_sequence, maxlen=char_max_length, padding="post")
    padded_char_sequence = pad_sequences([char_sequence], maxlen=max_length, padding='post', dtype='int32')
    return padded_word_sequence, padded_char_sequence

def predict_bad_words(sentence):
    max_length = 475
    padded_word_sequence, padded_char_sequence = preprocess_sentence(sentence)
    predictions = model_2.predict([padded_word_sequence, padded_char_sequence])  # Feed both sequences
    threshold = 0.5
    predicted_labels = (predictions > threshold).astype(int)[0]
    words = sentence.split()
    bad_words = [word for i, word in enumerate(words[:max_length]) if predicted_labels[i] == 1]
    return bad_words

def convert_mp3_to_wav(mp3_file, wav_file):
    try:
        subprocess.run(['ffmpeg', '-i', mp3_file, '-ac', '1', '-ar', '16000', wav_file], check=True)
    except subprocess.CalledProcessError as e:
        print("Error during MP3 to WAV conversion:", e)

def transcribe_audio_with_timestamps(audio_file, model_path):
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return []
    
    model = Model(model_path)
    if not audio_file.lower().endswith('.wav'):
        wav_file = audio_file.rsplit('.', 1)[0] + '.wav'
        if audio_file.lower().endswith('.mp3'):
            convert_mp3_to_wav(audio_file, wav_file)
            audio_file = wav_file
        else:
            print("Audio file must be in WAV format or MP3 format.")
            return []
    
    try:
        wf = wave.open(audio_file, "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
            print("Audio file must be WAV format mono PCM.")
            return []
    except Exception as e:
        print(f"Error opening audio file: {e}")
        return []
    
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)
    results = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            results.append(result)
        else:
            rec.PartialResult()
    final_result = json.loads(rec.FinalResult())
    results.append(final_result)
    
    word_timestamps = []
    for result in results:
        if 'result' in result:
            for word_info in result['result']:
                word_timestamps.append({
                    'word': word_info.get('word', ''),
                    'start': word_info.get('start', 0),
                    'end': word_info.get('end', 0)
                })
    return word_timestamps

def mute_bad_words_in_audio(audio_file, bad_words, word_timestamps):
    if not os.path.exists(audio_file):
        print(f"Audio file {audio_file} not found.")
        return None
    
    audio = AudioSegment.from_wav(audio_file)
    for item in word_timestamps:
        if item['word'] in bad_words:
            start_ms = item['start'] * 1000
            end_ms = item['end'] * 1000
            audio = audio[:start_ms] + AudioSegment.silent(duration=(end_ms - start_ms)) + audio[end_ms:]
    return audio

def process_audio_text(audio_file, model_path):
    try:
        # Step 1: Transcribe audio with timestamps
        word_timestamps = transcribe_audio_with_timestamps(audio_file, model_path)
        if not word_timestamps:
            raise ValueError("No transcribed words found in the audio.")

    except Exception as e:
        print(f"Error in transcription step: {e}")
        return None  # Or handle the error as needed

    try:
        # Step 2: Detect bad words from transcribed text
        transcribed_text = " ".join([item['word'] for item in word_timestamps])
        bad_words = predict_bad_words(transcribed_text)
        if not bad_words:
            print("No bad words detected in the transcription.")
            return audio_file  # Return original file if no bad words are detected

    except Exception as e:
        print(f"Error in bad word detection: {e}")
        return None  # Or handle the error as needed

    try:
        # Step 3: Mute bad words in audio
        muted_file = mute_bad_words_in_audio(audio_file, bad_words, word_timestamps)
        if not muted_file:
            raise ValueError("Failed to create muted audio file.")

    except Exception as e:
        print(f"Error in muting bad words: {e}")
        return None  # Or handle the error as needed

    try:
        # Save the processed (muted) audio to Azure Blob Storage
        processed_audio_path = f"muted_audio_{audio_file_name}"
        processed_audio_buffer = io.BytesIO()
        muted_file.export(processed_audio_buffer, format="wav")
        processed_audio_buffer.seek(0)

        blob_client = container_client.get_blob_client(processed_audio_path)
        blob_client.upload_blob(processed_audio_buffer, overwrite=True)

        return processed_audio_path
    except Exception as e:
        print(f"Error uploading to Azure: {e}")
        return None


@app.route("/process_audio", methods=["POST"])
def process_audio_route():
    try:
        # Get the file from the request
        file = request.files['audio_file']
        audio_file_name = file.filename

        # Save the uploaded file to a temporary location
        temp_path = f"temp_{audio_file_name}"
        file.save(temp_path)

        # Process the audio and mute bad words
        processed_audio_path = process_audio(temp_path)

        # If processing was successful, return the path of the processed file
        if processed_audio_path:
            return jsonify({"processed_audio_path": processed_audio_path})
        else:
            return jsonify({"error": "Failed to process audio"}), 500

    except Exception as e:
        print(f"Error in /process_audio route: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if PORT is not set
    app.run(debug=True, host="0.0.0.0", port=port)
