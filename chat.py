from flask import Flask, request, jsonify, send_file
from googletrans import Translator
from flask_cors import CORS
import pyttsx3
import io

app = Flask(__name__)
CORS(app)  # allow cross-origin requests from your HTML

translator = Translator()
tts_engine = pyttsx3.init()

@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"translation": ""})
    try:
        translated = translator.translate(text, dest="en")
        return jsonify({"translation": translated.text})
    except Exception as e:
        print(e)
        return jsonify({"translation": ""})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message", "")
    # Simple chatbot: echo or predefined responses
    response = f"You said: {message}" if message else "---"
    return jsonify({"response": response})

@app.route("/tts", methods=["POST"])
def tts():
    data = request.get_json()
    text = data.get("text", "")
    lang = data.get("lang", "en")

    # Using pyttsx3 for simplicity (offline)
    tts_engine.setProperty('rate', 150)
    tts_engine.setProperty('voice', 'english')
    
    audio_file = io.BytesIO()
    tts_engine.save_to_file(text, "output.mp3")
    tts_engine.runAndWait()
    return send_file("output.mp3", mimetype="audio/mpeg")

if __name__ == "__main__":
    app.run(debug=True)
