#!/bin/bash

# Update package list and install necessary dependencies
apt-get update
apt-get install -y ffmpeg  # Install FFmpeg if needed

# Install any Python dependencies that might not be installed yet
pip install -r requirements.txt

# Run your Flask app using Gunicorn (adjust the app name as necessary)
gunicorn --bind 0.0.0.0:$PORT app:app
