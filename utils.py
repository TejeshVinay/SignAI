import cv2
import numpy as np
import base64
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque
import time

# ===== LABELS =====
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

SIGN_LABELS = [
    'A', 'B', 'Bye', 'C', 'D', 'E', 'F', 'G', 'H', 'Hello', 'I', 'ILoveYou', 'J', 'K', 'L',
    'M', 'Meet', 'N', 'No', 'O', 'P', 'Please', 'Q', 'R', 'S', 'T', 'Tell', 'Thankyou',
    'U', 'V', 'W', 'X', 'Y', 'Yes', 'Z', 'del', 'space'
]

WORD_LABELS = {"Bye", "Hello", "ILoveYou", "Meet", "No", "Please", "Tell", "Thankyou", "Yes"}

# ===== THRESHOLDS / TIMING =====
SIGN_THRESHOLD = 0.10
EMOTION_THRESHOLD = 0.10
REPEAT_COOLDOWN = 1.0

# ===== LOAD MODELS =====
emotion_model = load_model("models/video_rebuilt.keras")
# sign_model = load_model("models/sign_model_landmark_mlp.h5")

# Warm up models with dummy data to avoid first-call lag
_dummy_face = np.zeros((1, 48, 48, 1), dtype=np.float32)
_dummy_hand = np.zeros((1, 63), dtype=np.float32)
emotion_model(_dummy_face, training=False)
# sign_model(_dummy_hand, training=False)

# ===== MEDIAPIPE (light configs) =====
mp_face = mp.solutions.face_detection
# mp_hands = mp.solutions.hands
# CHANGE 1: Removed mp_draw import — no backend drawing needed

face_detector = mp_face.FaceDetection(
    min_detection_confidence=0.5,
    model_selection=0
)



# ===== SMOOTHING =====
# sign_window = deque(maxlen=5)
emotion_window = deque(maxlen=5)

# ===== FRAME SIZE FOR PROCESSING =====
FRAME_WIDTH = 480
FRAME_HEIGHT = 360

# ===== SHARED STATE =====
# CHANGE 2: Removed "cap" from state — browser owns camera now
state = {
    "camera_running": False,
    "latest_sign": "None",
    "latest_sign_confidence": 0.0,
    "latest_emotion": "Unknown",
    "latest_emotion_confidence": 0.0,
    "current_sentence": "",
    "last_appended_sign": None,
    "last_append_time": 0.0,
    "frame_count": 0,
}

# Pre-allocate reusable array for landmarks
_keypoints_buffer = np.zeros(63, dtype=np.float32)




def start_camera():
    """
    CHANGE 3: This no longer opens hardware camera.
    It only resets all session state for a fresh detection session.
    Called automatically on first /predict request.
    """
    state["camera_running"] = True
    state["latest_sign"] = "None"
    state["latest_sign_confidence"] = 0.0
    state["latest_emotion"] = "Unknown"
    state["latest_emotion_confidence"] = 0.0
    state["current_sentence"] = ""
    state["last_appended_sign"] = None
    state["last_append_time"] = 0.0
    state["frame_count"] = 0
    # sign_window.clear()
    emotion_window.clear()


def stop_camera():
    """
    CHANGE 4: Only marks session as stopped.
    Does NOT clear sentence or predictions.
    Frontend keeps displaying last results after stop.
    Full reset happens on next start_camera() call.
    """
    state["camera_running"] = False


def safe_pop_last_char(text):
    if not text:
        return text
    return text[:-1]


def append_to_sentence(label):
    if label == "space":
        state["current_sentence"] += " "
    elif label == "del":
        state["current_sentence"] = safe_pop_last_char(state["current_sentence"])
    elif label in WORD_LABELS:
        if state["current_sentence"] and not state["current_sentence"].endswith(" "):
            state["current_sentence"] += " "
        state["current_sentence"] += f"{label} "
    else:
        state["current_sentence"] += label


def maybe_append_sign(stable_label):
    now = time.time()

    if stable_label is None or stable_label == "None":
        return

    last_sign = state["last_appended_sign"]
    last_time = state["last_append_time"]

    if last_sign != stable_label:
        append_to_sentence(stable_label)
        state["last_appended_sign"] = stable_label
        state["last_append_time"] = now
        return

    if now - last_time >= REPEAT_COOLDOWN:
        append_to_sentence(stable_label)
        state["last_appended_sign"] = stable_label
        state["last_append_time"] = now


def predict_emotion(frame, rgb):
    """
    CHANGE 5: Removed all drawing code.
    No cv2.rectangle, no cv2.putText.
    Only runs face detection + emotion model + updates state.
    """
    h, w, _ = frame.shape
    emotion_text = "Unknown"
    emotion_conf = 0.0

    face_results = face_detector.process(rgb)

    if face_results.detections:
        for det in face_results.detections:
            bbox = det.location_data.relative_bounding_box
            x = max(0, int(bbox.xmin * w))
            y = max(0, int(bbox.ymin * h))
            bw = max(1, int(bbox.width * w))
            bh = max(1, int(bbox.height * h))

            face = frame[y:y + bh, x:x + bw]
            if face.size == 0:
                continue

            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            gray = cv2.resize(gray, (48, 48))
            gray = gray.astype(np.float32) / 255.0
            gray = gray.reshape(1, 48, 48, 1)

            pred = emotion_model(gray, training=False)[0].numpy()
            cls = int(np.argmax(pred))
            conf = float(pred[cls])

            if conf >= EMOTION_THRESHOLD:
                emotion_window.append(cls)
                stable_cls = max(set(emotion_window), key=emotion_window.count)
                emotion_text = EMOTIONS[stable_cls]
                emotion_conf = conf
            else:
                emotion_text = "Unknown"
                emotion_conf = conf
                emotion_window.clear()

            break

    state["latest_emotion"] = emotion_text
    state["latest_emotion_confidence"] = emotion_conf


def predict_sign(frame, rgb):
    """
    CHANGE 6: Removed all drawing code.
    No mp_draw.draw_landmarks, no cv2.putText.
    Only runs hand detection + sign model + updates state.
    """
    sign_text = "None"
    sign_conf = 0.0

    hand_results = hands.process(rgb)

    if hand_results.multi_hand_landmarks:
        for hand in hand_results.multi_hand_landmarks:
            for i, lm in enumerate(hand.landmark):
                _keypoints_buffer[i * 3] = lm.x
                _keypoints_buffer[i * 3 + 1] = lm.y
                _keypoints_buffer[i * 3 + 2] = lm.z

            normalized = normalize_landmarks(_keypoints_buffer.copy())

            pred = sign_model(normalized.reshape(1, 63), training=False)[0].numpy()
            cls = int(np.argmax(pred))
            conf = float(pred[cls])

            if conf >= SIGN_THRESHOLD:
                sign_window.append(cls)
                stable_cls = max(set(sign_window), key=sign_window.count)
                sign_text = SIGN_LABELS[stable_cls]
                sign_conf = conf
                maybe_append_sign(sign_text)
            else:
                sign_text = "None"
                sign_conf = conf
                sign_window.clear()

            break

    state["latest_sign"] = sign_text
    state["latest_sign_confidence"] = sign_conf



def process_frame(frame):
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    state["frame_count"] += 1
    predict_emotion(frame, rgb)  # sign removed, only emotion

def process_frame_from_base64(frame_b64):
    """
    Called by /predict endpoint.
    Decodes base64 JPEG from frontend, runs both models, returns results.
    Auto-starts session on first call.
    """
    if not state["camera_running"]:
        start_camera()

    try:
        if "," in frame_b64:
            frame_b64 = frame_b64.split(",", 1)[1]

        img_bytes = base64.b64decode(frame_b64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is not None:
            process_frame(frame)

    except Exception as e:
        print(f"process_frame_from_base64 error: {e}")

    return {
        "sign": state["latest_sign"],
        "sign_confidence": round(state["latest_sign_confidence"], 4),
        "emotion": state["latest_emotion"],
        "emotion_confidence": round(state["latest_emotion_confidence"], 4),
        "sentence": state["current_sentence"]
    }