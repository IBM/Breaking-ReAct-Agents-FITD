import os
from utils import get_response_text,is_debug
from consts import *
from genai.client import Client
from genai.credentials import Credentials
from genai.schema import (DecodingMethod, HumanMessage, SystemMessage,
                          TextGenerationParameters)

import openai
from typing import List, Dict
import requests
import json
from openai import AzureOpenAI

class BaseModel:
    def __init__(self):
        self.model = None

    def prepare_input(self, sys_prompt,  user_prompt_filled):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def call_model(self, model_input):
        raise NotImplementedError("This method should be overridden by subclasses.")
 
class Bam_API_Model(BaseModel):
    def __init__(self, params,seed:int=DEAFULT_SEED,debug:bool=True):
        super().__init__()
        load_dotenv()
        assert os.getenv("GENAI_KEY") is not None, "Please set the GENAI_KEY environment variable"
        assert os.getenv("GENAI_API") is not None, "Please set the GENAI_API environment variable"
        if is_debug():
            print(f"Using BAM API model with key: {os.getenv('GENAI_KEY')}")
        self.debug=debug
        if not debug:
            self.client = Client(credentials=Credentials.from_env())
            self.parameters = TextGenerationParameters(
                decoding_method=DecodingMethod.SAMPLE, max_new_tokens=1000, min_new_tokens=30, temperature=0.7, top_k=50, top_p=1,random_seed=seed
                )
            self.params = params
        else:
            print('~~~~~~~~~~~~~ debug ~~~~~~~~~~~~~~~~~~`')
    def prepare_input(self, sys_prompt, user_prompt_filled):
        return {
            "sys_prompt": f"{sys_prompt}",
            "user_prompt": f"{user_prompt_filled}"
        }
    
    def call_model(self, input):
        if not self.debug:
            response = self.client.text.chat.create(
                model_id=self.params['model_name'],
                messages=[
                    SystemMessage(
                        content=input['sys_prompt'],
                    ),
                    HumanMessage(content=input['user_prompt']),
                ],
                parameters=self.parameters,
            )

            return response.results[0].generated_text
        else:
            return '~~~~~~~~~~~~~ debug ~~~~~~~~~~~~~~~~~~'

class LlamaModel(BaseModel):     
    def __init__(self, params):
        super().__init__()  
        import transformers
        import torch
        tokenizer = transformers.AutoTokenizer.from_pretrained(params['model_name'])
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=params['model_name'],
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_length=4096,
            do_sample=False, # temperature=0
            eos_token_id=tokenizer.eos_token_id,
        )

    def prepare_input(self, sys_prompt, user_prompt_filled):
        model_input = f"[INST] <<SYS>>\n{sys_prompt}\n<</SYS>>\n\n{user_prompt_filled} [/INST]"
        return model_input

    def call_model(self, model_input):
        output = self.pipeline(model_input)
        return get_response_text(output, "[/INST]")

class GPTModel(BaseModel):
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name['model_name']
        api_key = os.getenv("OPENAI_API_KEY")
        assert api_key is not None, "Please set the OPENAI_API_KEY environment variable"
        openai.api_key = api_key

    def prepare_input(self, sys_prompt: str, user_prompt_filled: str)->List[Dict]:
        return [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt_filled}]
    
    def call_model(self, model_input: dict):
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=model_input,
            temperature=0.7,
            stream=False
        )
        return response.choices[0].message.content



class AzureGPTModel:
    def __init__(self, params={'model_name': "o1-mini"}):
        super().__init__()
        self.model_name = params['model_name']
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = params.get('api_version',"2024-08-01-preview")
        self.endpoint =f"https://eteopenai.azure-api.net/openai/deployments/{self.model_name}-2024-09-12/chat/completions?api-version={self.api_version}" #currently for o1

        assert self.api_key is not None, "Please set the AZURE_OPENAI_API_KEY environment variable"

        print(f"AzureGPTModel initialized with model_name: {self.model_name}")

    def prepare_input(self, sys_prompt: str, user_prompt_filled: str) -> List[Dict]:
        return [
            {"role": "user", "content": sys_prompt},
            {"role": "user", "content": user_prompt_filled}
        ]

    def call_model(self, model_input: list):
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        data = {
            "messages": model_input
        }

        try:
            response = requests.post(self.endpoint, headers=headers, data=json.dumps(data))
            response.raise_for_status() 
            response_json = response.json()
            return response_json["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            print(f"Error making the API call: {e}")
            return None



MODELS = {
    "Llama": LlamaModel,
    "BAM_API": Bam_API_Model,
    "GPT": GPTModel,
    'AzureGPTModel': AzureGPTModel
}   