import os
from flask import Flask, request, jsonify
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import torch
from gpt4all_utils import ask_gpt4all

app = Flask(__name__)

model_name = "microsoft/resnet-50"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

def analyze_image_hf(image_path):
    image = Image.open(image_path)
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    labels = model.config.id2label
    return labels[predicted_class_idx]

@app.route('/')
def home():
    return "<h1>Agri AI Chatbot API Running</h1><p>Use POST /chat and /analyze-image endpoints.</p>"

@app.route('/chat', methods=['POST'])
def chat():
    question = request.json.get('question')
    if not question:
        return jsonify({'error': 'Question is required'}), 400
    try:
        answer = ask_gpt4all(question)
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    image_file = request.files.get('image')
    if not image_file:
        return jsonify({'error': 'Image file is required'}), 400

    image_path = "./temp_image.jpg"
    image_file.save(image_path)

    predicted_label = analyze_image_hf(image_path)
    return jsonify({'label': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
