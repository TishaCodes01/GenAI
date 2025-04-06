import requests

url = "http://127.0.0.1:8000/chat"
headers = {"Content-Type": "text/plain"}
data = "What are the symptoms of diabetes?"

response = requests.post(url, headers=headers, data=data)
print(response.text)
