from pprint import pprint
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file in the same directory
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, '.env')
load_dotenv(env_path)

BASE_URL = os.getenv('FPT_BASE_URL')
API_KEY = os.getenv('FPT_API_KEY')
MODEL_NAME = 'Llama-3.3-70B-Instruct'

client = OpenAI(api_key=API_KEY, 
                base_url=BASE_URL)

# This class instantiate the API, used to communicate with GPT
class LLAMA3_FPT:
    def __init__(self, method, sysprompt, example):
        self.id = 0
        self.chat_history = [
            {"role": "system", "content": sysprompt}
        ]
        if example:
            self.prompt = sysprompt + f'\nFollow these examples delimited with “”" as a guide.\n'
            keys = list(example.keys())
            for key in keys:
                input = example[key]
                index = input.find("\n") + 1
                self.prompt += f'“”"\nUser: {input[:index - 1]}\nAssistant: {input[index:]}“”"\n'
                self.chat_history.append({"role": "user", "content": input[:index - 1]})
                self.chat_history.append({"role": "assistant", "content": input[index:]})
        else:
            self.prompt = sysprompt

        
    def ask(self, prompt, max_tokens=1000):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            top_p=0.9,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    
    def chat(self, query, prompt="", stop=["\n"], max_tokens=2048):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": query},
                {"role": "assistant", "content": prompt}
            ],
            temperature=0,
            max_tokens=max_tokens,
            stop=stop
        )
        return response["choices"][0]["message"]["content"]
    
    def chat_with_image(self, chat_history, stop=["\n"], max_tokens=100):
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=chat_history,
            temperature=0
        )
        return response["choices"][0]["message"]["content"]