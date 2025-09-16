# server.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from googletrans import Translator
import requests
import time
import urllib.parse
import re
import json

app = Flask(__name__)
CORS(app)

translator = Translator(timeout=5)

# =======================
# Load crop names JSON
# =======================
with open("crop_names.json", "r", encoding="utf-8") as f:
    crop_names_dict = json.load(f)

# Flatten the nested crop_names dictionary if necessary
# (ensure all keys map to English values)
CROP_GLOSSARY = crop_names_dict  # assuming JSON already flat: Marathi/Punjabi/Hindi -> English

# =======================
# Existing domain glossary
# =======================
GLOSSARY = {
    "कांद्याचे पीक": "onion crop",
    "कांदा": "onion",
    "खत": "fertilizer",
    "खते": "fertilizers",
    "वापरू": "use",
    "कोणते कोणते": "which fertilizers",
    "जास्त नको बोलूस": "Don't talk too much",
    "तुम्ही काय करता": "What do you do",
    "तुम्ही काय म्हणता": "What are you saying",
}

# Merge crop glossary into domain glossary
GLOSSARY.update(CROP_GLOSSARY)

# Helper: replace longest keys first to avoid partial matches
GLOSSARY_KEYS_SORTED = sorted(GLOSSARY.keys(), key=lambda x: len(x), reverse=True)

# =======================
# Helper functions
# =======================
def pre_replace_glossary(text):
    """Replace known native-language phrases with their English terms to guide MT."""
    out = text
    for key in GLOSSARY_KEYS_SORTED:
        if key in out:
            out = out.replace(key, GLOSSARY[key])
    return out

def call_googletrans(text, src_lang):
    """Try googletrans (may be flaky)."""
    if src_lang == "auto" or not src_lang:
        res = translator.translate(text, dest="en")
    else:
        short = src_lang.split("-")[0]
        try:
            res = translator.translate(text, src=short, dest="en")
        except Exception:
            res = translator.translate(text, dest="en")
    return res.text

def call_mymemory(text, src_lang):
    """Call MyMemory free API as fallback. src_lang should be short code like 'mr' or 'pa'."""
    try:
        params = {
            'q': text,
            'langpair': f"{src_lang}|en"
        }
        r = requests.get("https://api.mymemory.translated.net/get", params=params, timeout=6)
        r.raise_for_status()
        j = r.json()
        return j.get("responseData", {}).get("translatedText", "")
    except Exception as e:
        raise

def call_libretranslate(text, src_lang):
    """Call LibreTranslate as another fallback (public instance)."""
    try:
        payload = {"q": text, "source": src_lang, "target": "en", "format": "text"}
        r = requests.post("https://libretranslate.de/translate", json=payload, timeout=6)
        r.raise_for_status()
        j = r.json()
        return j.get("translatedText", "")
    except Exception as e:
        raise

def contains_indic_script(s):
    return bool(re.search(r'[\u0900-\u097F\u0A00-\u0A7F\u0B80-\u0BFF\u0C00-\u0C7F\u0D00-\u0D7F]', s))

def is_reasonable_translation(translated, original):
    if not translated or len(translated.strip()) < 2:
        return False
    if contains_indic_script(translated):
        return False
    if translated.strip() == original.strip():
        return False
    return True

def ensure_glossary_terms_in_output(original, translated):
    out = translated
    for key, eng in GLOSSARY.items():
        if key in original:
            if eng.lower() not in out.lower():
                out_lower = out.lower()
                if "vegetable" in out_lower:
                    out = re.sub(r'vegetables?', eng, out, flags=re.IGNORECASE)
                else:
                    out = f"{out} ({eng})"
    return out

# =======================
# Flask routes
# =======================
@app.route("/", methods=["GET"])
def home():
    try:
        return render_template("index.html")
    except Exception:
        return "Index page (place index.html in templates/ to serve from Flask)", 200

@app.route("/translate", methods=["POST"])
def translate_api():
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get("text") or "").strip()
    user_lang = data.get("lang") or "auto"

    if not text:
        return jsonify({"error": "No text provided"}), 400

    pre_text = pre_replace_glossary(text)

    detected = None
    try:
        detected = translator.detect(text).lang
    except Exception:
        detected = user_lang if user_lang else "auto"

    src_lang = detected or user_lang or "auto"
    short_src = src_lang.split("-")[0] if src_lang and "-" in src_lang else src_lang

    services_used = []
    translated = None

    for attempt in range(2):
        try:
            gt = call_googletrans(pre_text, short_src if short_src else "auto")
            services_used.append("googletrans")
            if is_reasonable_translation(gt, text):
                translated = gt
                break
        except Exception:
            time.sleep(0.6)
            continue

    if not translated:
        try:
            mm = call_mymemory(pre_text, short_src if short_src else "auto")
            services_used.append("mymemory")
            if is_reasonable_translation(mm, text):
                translated = mm
        except Exception:
            pass

    if not translated:
        try:
            lb = call_libretranslate(pre_text, short_src if short_src else "auto")
            services_used.append("libretranslate")
            if is_reasonable_translation(lb, text):
                translated = lb
        except Exception:
            pass

    if not translated:
        try:
            gt2 = call_googletrans(text, short_src if short_src else "auto")
            services_used.append("googletrans-fallback")
            if is_reasonable_translation(gt2, text):
                translated = gt2
        except Exception:
            pass

    if not translated:
        translated = text
        services_used.append("original-fallback")

    translated = ensure_glossary_terms_in_output(text, translated)

    response = {
        "originalText": text,
        "preProcessedText": pre_text,
        "detectedLang": src_lang,
        "translatedText": translated,
        "servicesTried": services_used
    }
    return jsonify(response), 200

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
