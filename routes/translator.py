from flask import Blueprint, request, jsonify, send_file
from deep_translator import GoogleTranslator
from gtts import gTTS
from langdetect import detect, DetectorFactory
import io
import json
import os

# Ensure consistent language detection
DetectorFactory.seed = 0

DATA_DIR = os.path.join(os.path.dirname(__file__), "../utils")
with open(os.path.join(DATA_DIR, "crop_names.json"), "r", encoding="utf-8") as f:
    crop_names = json.load(f)

languages = ["marathi", "malayalam", "hindi", "punjabi", "tamil", "telugu"]
grammar_data = {}
for lang in languages:
    path = os.path.join(DATA_DIR, f"{lang}_grammar.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            grammar_data[lang] = json.load(f)

translator_bp = Blueprint("translator_bp", __name__)

def translate_crop(text):
    return crop_names.get(text, text)

@translator_bp.route("/translate", methods=["POST"])
def translate_text():
    data = request.get_json()
    text = data.get("text", "").strip()
    print("Received text:", text)  # DEBUG

    if not text:
        print("No text provided")
        return jsonify({"translation": ""})

    # Check crop names first
    translated_text = crop_names.get(text, text)
    print("After crop check:", translated_text)  # DEBUG

    # Google Translate fallback
    if translated_text == text:
        try:
            translated_text = GoogleTranslator(source='auto', target='en').translate(text)
            print("After GoogleTranslator:", translated_text)  # DEBUG
        except Exception as e:
            print("Translation error:", e)
            translated_text = ""

    return jsonify({"translation": translated_text})


@translator_bp.route("/tts", methods=["POST"])
def tts():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Detect language automatically
    try:
        lang_detected = detect(text)
    except:
        lang_detected = "en"  # fallback

    # Generate TTS audio in memory
    tts = gTTS(text=text, lang=lang_detected)
    audio_bytes = io.BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)

    return send_file(audio_bytes, mimetype="audio/mpeg")
