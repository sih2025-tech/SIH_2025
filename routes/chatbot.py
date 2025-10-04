from flask import Blueprint, request, jsonify

chatbot_bp = Blueprint('chatbot', __name__)

@chatbot_bp.route('/ask', methods=['POST'])
def ask_bot():
    data = request.json
    message = data.get('message')

    # Dummy response for now
    response = f"Echo: {message}"
    return jsonify({"response": response})
