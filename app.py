from flask import Flask, render_template, request, jsonify
from utils import state, process_frame_from_base64, stop_camera
from sentence_generator import SentenceGenerator

app = Flask(__name__, template_folder="templates", static_folder="static")
generator = SentenceGenerator()

@app.route("/")
def home():
    return render_template("index.html")


# CHANGE 1: No /video_feed route — browser owns webcam now

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "frame" not in data:
        return jsonify({"error": "No frame provided"}), 400

    result = process_frame_from_base64(data["frame"])
    return jsonify(result), 200


@app.route("/generate_sentence", methods=["POST"])
def generate_sentence():
    try:
        data = request.get_json()

        gestures = data.get("gestures", [])
        emotion = data.get("emotion", "neutral")

        # Call your Groq-powered generator
        result = generator.generate(gestures, emotion)

        return jsonify(result)

    except Exception as e:
        print("Error in generate_sentence:", e)
        return jsonify({
            "error": "Something went wrong",
            "sentence": None
        }), 500


# CHANGE 2: Accept both GET and POST for sendBeacon compatibility
@app.route("/stop_camera", methods=["GET", "POST"])
def stop_camera_route():
    stop_camera()
    return jsonify({"status": "stopped"})


if __name__ == "__main__":
    # CHANGE 3: debug=False and threaded=True for performance
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)