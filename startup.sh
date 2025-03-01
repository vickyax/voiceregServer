#!/bin/bash

# Install system dependencies
apt-get update && apt-get install -y libsndfile1 ffmpeg

# Start Gunicorn
gunicorn --bind 0.0.0.0:8000 --timeout 600 app:app