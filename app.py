from flask import Flask
#testing CORS
from flask_cors import CORS
from routes.weather import weather_bp
from routes.market import market_bp
from routes.translator import translator_bp
import ee

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize Earth Engine with service account
SERVICE_ACCOUNT_FILE = "service_account.json"
SERVICE_ACCOUNT_EMAIL = "weather-predic-acc@weather4-470605.iam.gserviceaccount.com"

credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT_EMAIL, SERVICE_ACCOUNT_FILE)
ee.Initialize(credentials)

# Register blueprints
app.register_blueprint(weather_bp, url_prefix='/weather')
app.register_blueprint(market_bp, url_prefix='/market')
app.register_blueprint(translator_bp, url_prefix="/translator")


if __name__ == "__main__":
    app.run(debug=True)


from flask import send_from_directory
import os

@app.route("/")
def serve_html():
    # Use absolute path to your BackEnd folder
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(base_dir, "test.html")