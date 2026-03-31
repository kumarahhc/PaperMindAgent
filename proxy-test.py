#!/usr/bin/python3

import requests
import json

# API Gateway Proxy URL - this forwards to https://openrouter.ai/api/v1
BASE_URL = "https://5f5832nb90.execute-api.eu-central-1.amazonaws.com/v1"

response = requests.post(
    f"{BASE_URL}/chat/completions",
    headers={"Content-Type": "application/json"},
    json={
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "user", "content": "What is the capital of Finland?"}
        ],
    },
)

print(f"Status: {response.status_code}")
print(json.dumps(response.json(), indent=2))


