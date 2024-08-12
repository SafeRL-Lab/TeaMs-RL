'''
  Code taken from https://github.com/google-research/google-research/blob/master/saycan/SayCan-Robot-Pick-Place.ipynb
  Licensed under Apache 2.0 license
'''

import openai
import os

from openai import OpenAI
client = OpenAI(api_key="your_openai_api_key")


def gpt3_call(prompt="How are you?"):    
     test_chat_message = [{"role": "user", "content": prompt}]
     response = client.chat.completions.create(
     model= "gpt-4-1106-preview", #"gpt-4-1106-preview", # "gpt-3.5-turbo-1106",    
     messages=test_chat_message,
     temperature=0.7,
     max_tokens=2048,
     top_p=0.95,
     frequency_penalty=0,
     presence_penalty=0
     )  
     return response.choices[0].message.content # response

def gpt3_call_count_actions(prompt="How are you?"):    
     
     return "Count policy actions and give action distribution" # response
   
   

