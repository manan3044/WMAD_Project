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

# Define the segment duration (500 milliseconds)
segment_duration = 500

# Azure Blob Storage connection
connection_string = "DefaultEndpointsProtocol=https;AccountName=wmadprojectcontainer;AccountKey=AVZbuCTn7xKm4Ere7oGSfvC3JTxuBca9j76yripqxok3J9xKiYiIomY6sX3qHFP0O5oTjoNkTJia+AStFGlfmg==;EndpointSuffix=core.windows.net"
container_name = "audio-files"
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
def process_audio(blob_client):
    try:
        bad_word_indices = []  # Store indices of bad word segments
        segments = []  # To store MFCC features for each segment

        # Download the audio file from Azure Blob Storage
        download_stream = blob_client.download_blob()
        audio = AudioSegment.from_file(io.BytesIO(download_stream.readall()))

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
            processed_audio_buffer = io.BytesIO()
            audio.export(processed_audio_buffer, format="wav")
            processed_audio_buffer.seek(0)

            processed_blob_name = "processed/muted_audio.wav"
            processed_blob_client = container_client.get_blob_client(processed_blob_name)
            processed_blob_client.upload_blob(processed_audio_buffer, overwrite=True)

            return processed_blob_name
        else:
            print("No valid segments to predict.")
            return None
    except Exception as e:
        print(f"Error processing audio file: {e}")
        return None

@app.route("/process_audio", methods=["POST"])
def process_audio_route():
    try:
        # Get the file from the request
        file = request.files['audio_file']
        audio_file_name = file.filename

        # Upload the received file to Azure Blob Storage
        blob_client = container_client.get_blob_client(audio_file_name)
        file.seek(0)
        blob_client.upload_blob(file, overwrite=True)

        # Process the audio and mute bad words
        processed_audio_path = process_audio(blob_client)

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
