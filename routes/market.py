from flask import Blueprint, jsonify, request
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException
from difflib import get_close_matches
import pandas as pd
import time

from .market_rates import crop_en_to_mr, city_en_to_mr

market_bp = Blueprint('market', __name__)

ESSENTIAL_COLS = ['बाजार समिती', 'किमान', 'जास्तीत जास्त', 'मॉडल']


def fetch_msamb_by_crop(crop_mr):
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")

    driver = webdriver.Chrome(options=options)
    try:
        driver.get("https://www.msamb.com/ApmcDetail/APMCPriceInformation")
        wait = WebDriverWait(driver, 15)

        # Select commodity
        try:
            select_elem = wait.until(EC.presence_of_element_located((By.ID, "drpCommodities")))
            Select(select_elem).select_by_visible_text(crop_mr)
        except NoSuchElementException:
            return pd.DataFrame()

        time.sleep(2)  # allow table to refresh

        rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
        if not rows:
            return pd.DataFrame()

        data = [[td.text.strip() for td in row.find_elements(By.TAG_NAME, "td")] for row in rows]
        if not data:
            return pd.DataFrame()

        filtered_data = [row[:len(ESSENTIAL_COLS)] for row in data]
        df = pd.DataFrame(filtered_data, columns=ESSENTIAL_COLS)
        return df

    finally:
        driver.quit()


@market_bp.route('/rates', methods=['GET'])
def get_market_rates():
    crop_en = request.args.get('crop', '').strip().lower()
    city_en = request.args.get('city', '').strip().lower()

    if not crop_en or not city_en:
        return jsonify({"error": "Please provide 'crop' and 'city' query parameters"}), 400

    crop_mr = crop_en_to_mr.get(crop_en)
    if not crop_mr:
        return jsonify({"error": f"Crop '{crop_en}' not found"}), 404

    df = fetch_msamb_by_crop(crop_mr)

    if df.empty:
        return jsonify({"error": "No market data found on MSAMB"}), 500

    # Fuzzy match for city
    city_keys = [key.lower() for key in city_en_to_mr.keys()]
    match = get_close_matches(city_en, city_keys, n=1, cutoff=0.1)

    if not match:
        return jsonify({"error": f"City '{city_en}' not found"}), 404

    city_key = next(key for key in city_en_to_mr if key.lower() == match[0])
    city_mr = city_en_to_mr[city_key]

    filtered = df[df['बाजार समिती'].str.contains(city_mr, case=False, na=False, regex=True)]

    if filtered.empty:
        markets = df['बाजार समिती'].dropna().tolist()
        best_match = get_close_matches(city_mr, markets, n=1, cutoff=0.1)
        if not best_match:
            return jsonify({"error": f"No market data found for city '{city_en}'"}), 404
        filtered = df[df['बाजार समिती'] == best_match[0]]

    if filtered.empty:
        return jsonify({"error": f"No market data found for city '{city_en}'"}), 404

    # ✅ Keep only rate change columns
    # Keep only the 'Modal' column
    rates_only = filtered[['मॉडल']].copy()

    # Rename to English
    rates_only.rename(columns={'मॉडल': 'Price'}, inplace=True)

    return jsonify(rates_only.to_dict(orient='records'))

