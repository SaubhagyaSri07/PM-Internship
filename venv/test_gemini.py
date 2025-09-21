import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("❌ No GEMINI_API_KEY found.")
    raise SystemExit(1)

print("Using key:", API_KEY[:6] + "*****")

# ✅ Notice v1beta here
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={API_KEY}"
data = {"contents": [{"parts": [{"text": "Say hello world in 5 words"}]}]}

try:
    r = requests.post(url, json=data, timeout=10)
    r.raise_for_status()
    print("✅ Gemini API response (truncated):")
    print(r.json())
except requests.exceptions.HTTPError as he:
    print("❌ HTTP error:", he, "Response:", getattr(he.response, "text", ""))
except Exception as e:
    print("❌ Error calling Gemini API:", e)
