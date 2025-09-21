import requests

GPT4ALL_API_URL = "http://localhost:4891/v1/chat/completions"

def ask_gpt4all(question, max_tokens=512):
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "gpt4all",
        "messages": [{"role": "user", "content": question}],
        "max_tokens": max_tokens
    }
    response = requests.post(GPT4ALL_API_URL, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        return result['choices'][0]['message']['content']
    else:
        raise Exception(f"GPT4All API Error: {response.status_code} - {response.text}")
