# Agri AI Project

## Setup Backend

1. Navigate to `backend`
2. Install dependencies: `pip install -r requirements.txt`
3. Run Flask app: `python app.py`
4. Ensure GPT4All API is running locally on port 4891

## Setup Frontend

1. Navigate to `frontend`
2. Install dependencies: `npm install`
3. Run React app: `npm start`
4. Open `http://localhost:3000` in your browser

## Description

- Backend serves AI chatbot text answering and image classification APIs.
- Frontend connects via REST to backend, allowing asking questions and image uploads.

## Notes

- Update backend URLs in frontend if your server is hosted differently.
- Keep GPT4All API and Flask backend running for full functionality.
