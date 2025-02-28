from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import librosa
import logging
from sklearn.mixture import GaussianMixture
import joblib
from werkzeug.utils import secure_filename
import re
import soundfile as sf
# Enable logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
CORS(app)

# Configuration
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac', 'ogg'}
SAMPLE_RATE = 16000  # Target sample rate

# Ensure required directories exist
os.makedirs('server/models', exist_ok=True)
os.makedirs('server/uploads', exist_ok=True)

# Load existing speaker database
speakers_db = {}
if os.path.exists('server/models/speakers_db.pkl'):
    speakers_db = joblib.load('server/models/speakers_db.pkl')
    logging.info(f"Loaded {len(speakers_db)} registered speakers.")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(audio_path, n_mfcc=13):
    """Extract MFCC features with Windows compatibility."""
    try:
        # Use soundfile as fallback for Windows
        try:
            y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True, res_type='kaiser_fast')
        except Exception as librosa_error:
            logging.warning(f"Librosa load failed: {librosa_error}. Trying soundfile...")
            y, sr = sf.read(audio_path)
            y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE

        # Ensure audio is mono
        if len(y.shape) > 1:
            y = librosa.to_mono(y)

        # Pad short audio files
        if len(y) < SAMPLE_RATE // 2:  # < 0.5 second
            y = np.pad(y, (0, max(0, SAMPLE_RATE // 2 - len(y))), mode='constant')

        # Extract features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)
        delta = librosa.feature.delta(mfccs)
        delta2 = librosa.feature.delta(mfccs, order=2)
        
        features = np.vstack([mfccs, delta, delta2]).T
        return features

    except Exception as e:
        logging.error(f"Feature extraction failed: {e}")
        return None

def validate_features(features):
    """Ensure features are suitable for training."""
    if features is None:
        return False
    if len(features) < 10:
        logging.error("Insufficient features (less than 10 frames)")
        return False
    if np.allclose(features, features[0]):  # Check for constant values
        logging.error("All feature vectors are identical")
        return False
    return True

def train_gmm(features, n_components=8):  # Reduced from 16 to 8
    """Train a Gaussian Mixture Model with regularization."""
    try:
        logging.debug(f"Training GMM with {len(features)} samples, shape: {features.shape}")
        
        # Add data validation
        if len(features) < n_components * 5:  # Minimum 5 samples per component
            logging.error(f"Insufficient samples ({len(features)}) for {n_components} components")
            return None
            
        # Add regularization and limit iterations
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type='diag',
            reg_covar=1e-2,  # Increased regularization
            max_iter=100,    # Reduced from 200
            n_init=3,        # Multiple initializations
            random_state=42
        )
        
        gmm.fit(features)
        return gmm
    except Exception as e:
        logging.error(f"GMM training error: {str(e)}")
        return None

@app.route('/register', methods=['POST'])

def register_speaker():
    try:
        if 'name' not in request.form or 'audio' not in request.files:
            return jsonify({'error': 'Speaker name and audio file required'}), 400

        speaker_name = re.sub(r'[^\w\-]', '_', request.form['name'])
        audio_file = request.files['audio']

        if not allowed_file(audio_file.filename):
            return jsonify({'error': 'Unsupported file format'}), 400

        filename = secure_filename(f"{speaker_name}_{audio_file.filename}")
        audio_path = os.path.join('server/uploads', filename)
        audio_file.save(audio_path)

        if not os.path.exists(audio_path):
            logging.error("File was not saved properly.")
            return jsonify({'error': 'File save failed'}), 500

        logging.debug(f"Audio saved at {audio_path}")

        features = extract_features(audio_path)

        

        if not validate_features(features):

            os.remove(audio_path)

            return jsonify({'error': 'Insufficient or invalid audio content'}), 400


        # Try different component counts if initial training fails

        for components in [8, 4, 2]:  # Fallback to simpler models

            gmm = train_gmm(features, n_components=components)

            if gmm:

                speakers_db[speaker_name] = gmm

                joblib.dump(speakers_db, 'server/models/speakers_db.pkl')

                os.remove(audio_path)

                return jsonify({'message': f'Registered with {components} components'}), 200

        

        os.remove(audio_path)

        return jsonify({'error': 'Failed to train viable model'}), 500

    except Exception as e:
        logging.error(f"Registration error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/recognize', methods=['POST'])
def recognize_speaker():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'Audio file required'}), 400
            
        if not speakers_db:
            return jsonify({'error': 'No registered speakers'}), 400

        audio_file = request.files['audio']
        if not allowed_file(audio_file.filename):
            return jsonify({'error': 'Unsupported file format'}), 400

        filename = secure_filename(f"recognize_{audio_file.filename}")
        audio_path = os.path.join('server/uploads', filename)
        audio_file.save(audio_path)

        logging.debug(f"Recognizing speaker from file: {audio_path}")

        features = extract_features(audio_path)
        os.remove(audio_path)
        
        if features is None:
            return jsonify({'error': 'Invalid audio content'}), 400

        scores = {}
        for name, model in speakers_db.items():
            try:
                score = np.mean(model.score_samples(features))
                scores[name] = score
                logging.debug(f"Score for {name}: {score}")
            except Exception as e:
                logging.error(f"Scoring error for {name}: {e}")
                continue

        if not scores:
            return jsonify({'error': 'Recognition failed'}), 400

        best_speaker, best_score = max(scores.items(), key=lambda x: x[1])

        # Compute confidence
        score_values = np.array(list(scores.values()))
        confidence = 100 * (best_score - score_values.min()) / (score_values.max() - score_values.min() + 1e-8)

        logging.info(f"Recognized speaker: {best_speaker} with confidence {confidence:.2f}%")

        return jsonify({
            'speaker': best_speaker,
            'confidence': round(float(confidence), 2)
        }), 200

    except Exception as e:
        logging.error(f"Recognition error: {e}")
        return jsonify({'error': 'Processing failed'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'registered_speakers': len(speakers_db),
        'audio_formats': list(ALLOWED_EXTENSIONS)
    }), 200

if __name__ == '__main__':
    logging.info("Starting Flask server on port 5000...")
    app.run()
