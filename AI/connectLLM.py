import requests
import json
from dotenv import load_dotenv
import os
# Load environment variables from .env file
load_dotenv()
class ConnectLLM:
    @staticmethod
    def senText(prompt):
        URL = os.getenv('URL')
        data = {
            "model": "gpt-3.5-turbo",
            "prompt": prompt,
            "stream" : False
        }
        response = requests.post(URL, data=json.dumps(data))
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.text}
    
    @staticmethod
    def senTextandImage(prompt, image):
        URL = os.getenv('URL')
        Model = os.getenv('MODEL')
        data = {
            "model": Model,
            "prompt": prompt,
            "image": image,
            "stream" : False
        }
        response = requests.post(URL, data=json.dumps(data))
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.text}