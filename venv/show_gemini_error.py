import os, requests, json
k = os.getenv("GEMINI_API_KEY")
print("Key present:", bool(k))
url = f"https://generativelanguage.googleapis.com/v1beta/models?key={k}"
try:
    r = requests.get(url, timeout=10)
    print("Status:", r.status_code)
    try:
        print("Body:", json.dumps(r.json(), indent=2))
    except Exception:
        print("Body (raw):", r.text)
except Exception as e:
    print("Exception calling models list:", e)