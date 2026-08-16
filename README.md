# SignAI - Multimodal Sign Language and Emotion Intelligence

SignAI is a web-based assistive communication system that detects sign language gestures and facial emotions in real time. It combines browser-based hand landmark detection, TensorFlow models, Flask APIs, and LLM-powered sentence generation to convert detected gestures into meaningful English sentences.

## Features

- Real-time sign language detection from webcam input
- Browser-side hand tracking using MediaPipe and TensorFlow.js
- Facial emotion recognition using a trained Keras model
- AI-powered sentence generation from detected gestures and emotion context
- Editable transcript and session history
- Text-to-speech support in the frontend
- Analytics and learning pages for ASL practice

## Tech Stack

- Python
- Flask
- TensorFlow / Keras
- TensorFlow.js
- MediaPipe
- OpenCV
- NumPy / Pandas / scikit-learn
- HTML, CSS, JavaScript
- Llama 3.1 for sentence generation

## Project Structure

```text
.
+-- app.py                         # Flask application entry point
+-- utils.py                       # Emotion processing and shared state helpers
+-- sentence_generator.py          # Groq-based sentence generation logic
+-- requirements.txt               # Python dependencies
+-- labels.json                    # Supported sign labels
+-- templates/
|   +-- index.html                 # Main web interface
+-- static/
|   +-- app.js                     # Frontend logic for detection, transcript, and UI
|   +-- styles.css                 # Application styling
|   +-- models/sign_model_tfjs/    # TensorFlow.js sign detection model
+-- models/                        # Keras models for sign/emotion detection
+-- newdata/                       # Sign image dataset
+-- MP_Data_Images/                # Landmark/keypoint dataset
+-- face-dataset/                  # Facial emotion dataset
```

## Supported Signs

The current sign model supports:

```text
A, B, Bye, C, D, E, F, G, H, Hello, I, ILoveYou, J, K, L, M, Meet,
N, No, O, P, Please, Q, R, S, T, Tell, Thankyou, U, V, W, X, Y,
Yes, Z, del, space
```

## Setup

1. Clone or download the project.

2. Create and activate a virtual environment.

```bash
python -m venv myenv
myenv\Scripts\activate
```

On macOS/Linux:

```bash
python -m venv myenv
source myenv/bin/activate
```

3. Install dependencies.

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your Groq API key.

```env
GROQ_APIKEY=your_groq_api_key_here
```

## Running the Application

Start the Flask server:

```bash
python app.py
```

Open the application in your browser:

```text
http://localhost:5000
```

Allow camera access when prompted. The browser uses the webcam for live hand tracking, while the backend receives frames for emotion recognition.

## API Endpoints

### `GET /`

Serves the main SignAI interface.

### `POST /predict`

Accepts a base64-encoded video frame and returns the latest emotion/sign state.

### `POST /generate_sentence`

Accepts detected gestures and emotion, then returns an AI-generated sentence.

Example request body:

```json
{
  "gestures": ["Hello", "Please"],
  "emotion": "Happy"
}
```

### `GET|POST /stop_camera`

Marks the current camera session as stopped on the backend.

## Notes

- The TensorFlow.js sign model is loaded from `static/models/sign_model_tfjs/model.json`.
- The backend emotion model is loaded from `models/video_rebuilt.keras`.
- Sentence generation requires a valid Groq API key in `.env`.
- Some frontend libraries are loaded from CDNs, so an internet connection may be required when opening the app.
- Large dataset folders are included for training and experimentation, but they are not required for simply running the web app if the trained models already exist.

## Future Improvements

- Add model training scripts and usage instructions
- Add automated tests for Flask endpoints
- Improve model confidence handling and calibration
- Add deployment instructions
- Add screenshots or demo video
