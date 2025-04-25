import requests
import json
from dotenv import load_dotenv
import os
# Load environment variables from .env file
load_dotenv()
class ConnectLLM:
    @staticmethod
    def senText(prompt, image=None, context=None):
        URL = os.getenv('URL')
        MODEL = os.getenv('MODEL')

        if context:
            system_prompt = f"""Previous relevant conversations:
{context}

Current question: {prompt}

Please provide a response that:
1. Uses the context when relevant
2. Provides accurate and up-to-date information
3. Is clear and concise"""
        else:
            system_prompt = prompt

        data = {
            "model": MODEL,
            "prompt": system_prompt,
            "stream": False,
            "images": [image] if image else []
        }
        
        print(f"Sending to LLM - Context: {'Yes' if context else 'No'}, Prompt length: {len(system_prompt)}")
        response = requests.post(URL, data=json.dumps(data))
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.text}