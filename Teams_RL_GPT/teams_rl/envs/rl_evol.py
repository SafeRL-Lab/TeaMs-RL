# depth action add constraints, deepening, concretizing, increase reasoning steps, and complicate input.
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer #, AdapterConfig
import loralib as lora
from peft import PeftModel
from pathlib import Path
import os
import sys

from .openai_api_3dot5_VT import gpt3_call


import json  
  



import json
import os
import time
from datetime import datetime
currentDateAndTime = datetime.now()
start_run_date_and_time=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())


folder = os.getcwd()[:-6] + 'data_VT/' + 'gpt-4-1106-preview_40k_41k_prompt_Alpaca'+ '/' + str(
        start_run_date_and_time) + '/'

if not os.path.exists(folder):
    os.makedirs(folder)


with open(folder + '/all_state.json', 'w') as f:
    f.writelines('------------------ start ------------------' + '\n')
with open(folder + '/prompt_state.json', 'w') as f:
    f.writelines('------------------ start ------------------' + '\n')

    # json.dump(state,f)
    # f.writelines('------------------- end -------------------')



print("start to generate prompt-------------------")

# instruction = "How to write an academic paper?"
class Evol_gpt():
    def __init__(self, **kwargs):
        self.action_space = ["add_constraints", "deepening", "concretizing", "increase_reasoning_steps", "complicate_input", "breadth_action"]
        self.observation_space = [0, 1, 2, 3, 4, 5]

    def reset(self):
        self.state = [0, 0, 0, 0, 0, 0]
        reward=0
        cost=0
        info = {
            "reward": reward,
            "cost": cost,
        }
        return self.state, info
    
    def read_json_file(self, file_path):
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []
        return data

    def write_json_file(self, file_path, data):
        with open(file_path, 'w') as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

    def append_to_json_file(self, file_path, new_data):
        data = self.read_json_file(file_path)
        data.append(new_data)
        self.write_json_file(file_path, data)
        

    def model(self, evol_prompt):          
        time.sleep(2)
        # state = generate_response_text_003(evol_prompt)  
        state = gpt3_call(evol_prompt)        
        # state = response.choices[0].message.content # state["choices"][0]["message"]["content"]

        # print("state--2----------:", state)
        
        file_path = folder + '/all_state.json'
        self.append_to_json_file(file_path=file_path, new_data=state)

        return state

    def action_take(self, action, instruction):
        if action == "breadth_action":
            evol_prompt = """I want you act as a Prompt Creator.
            Your goal is to draw inspiration from the #Given Prompt# to create a brand new prompt.
            This new prompt should belong to the same domain as the #Given Prompt# but be even more rare.
            The LENGTH and difficulty level of the #Created Prompt# should be similar to that of the #Given Prompt#. 
            Don't repeat the conditions and requirements in the response, and Don't disclose your role.
            The Prompt Rewriter Must not give the introduction and explain the reason, the Prompt Rewriter must just give the most relevant response.
            This new prompt should not exceed 2048 words.
            The #Created Prompt# must be reasonable and must be understood and responded by humans.
            ‘#Given Prompt#’, ‘#Created Prompt#’, ‘given prompt’ and ‘created prompt’ are not allowed to appear in #Created Prompt#. 
            #Given Prompt#:
            """ + instruction
            created_prompt = self.model(evol_prompt)
            state_encode = [1,0,0,0,0,0]
            cost = 1
            return created_prompt, state_encode, cost
        elif action == "add_constraints":
            evol_prompt = """
            I want you act as a Prompt Rewriter.
            Your objective is to rewrite a given prompt into a more complex version to make those famous AI systems (e.g., chatgpt and GPT4) a bit harder to handle. 
            But the rewritten prompt must be reasonable and must be understood and responded by humans. 
            Your rewriting cannot omit the non-text parts such as the table and code in #Given Prompt#:. Also, please do not omit the input in #Given Prompt#. 
            Don't repeat the conditions and requirements in the response, and Don't disclose your role.
            The Prompt Rewriter Must not give the introduction and explain the reason, the Prompt Rewriter must just give the most relevant response.
            This new prompt should not exceed 2048 words.
            You SHOULD complicate the given prompt using the following method: 
            Please add one more constraints/requirements into #Given Prompt#  
            You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add or replace 10 to 20 words into #Given Prompt#.
            ‘#Given Prompt#’, ‘#Rewritten Prompt#’, ‘given prompt’ and ‘rewritten prompt’ are not allowed to appear in #Rewritten Prompt#.  
            #Given Prompt#:
            """ + instruction
            created_prompt = self.model(evol_prompt)
            state_encode = [0,1,0,0,0,0]
            cost = 1
            return created_prompt, state_encode, cost
        elif action == "deepening":
            evol_prompt = """
            I want you act as a Prompt Rewriter.
            Your objective is to rewrite a given prompt into a more complex version to make those famous AI systems (e.g., chatgpt and GPT4) a bit harder to handle. 
            But the rewritten prompt must be reasonable and must be understood and responded by humans. 
            Your rewriting cannot omit the non-text parts such as the table and code in #Given Prompt#:. Also, please do not omit the input in #Given Prompt#.
            Don't repeat the conditions and requirements in the response, and Don't disclose your role.
            The Prompt Rewriter Must not give the introduction and explain the reason, the Prompt Rewriter must just give the most relevant response.
            This new prompt should not exceed 2048 words.
            You SHOULD complicate the given prompt using the following method:          
            If #Given Prompt# contains inquiries about certain issues, the depth and breadth of the inquiry can be increased. 
            You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #Given Prompt#.
            ‘#Given Prompt#’, ‘#Rewritten Prompt#’, ‘given prompt’ and ‘rewritten prompt’ are not allowed to appear in #Rewritten Prompt#.  
            #Given Prompt#:
            """ + instruction
            created_prompt = self.model(evol_prompt)
            state_encode = [0,0,1,0,0,0]
            cost = 1
            return created_prompt, state_encode, cost
        elif action == "concretizing":
            evol_prompt = """
            I want you act as a Prompt Rewriter.
            Your objective is to rewrite a given prompt into a more complex version to make those famous AI systems (e.g., chatgpt and GPT4) a bit harder to handle. 
            But the rewritten prompt must be reasonable and must be understood and responded by humans. 
            Your rewriting cannot omit the non-text parts such as the table and code in #Given Prompt#:. Also, please do not omit the input in #Given Prompt#.
            Don't repeat the conditions and requirements in the response, and Don't disclose your role.
            The Prompt Rewriter Must not give the introduction and explain the reason, the Prompt Rewriter must just give the most relevant response.
            This new prompt should not exceed 2048 words.
            You SHOULD complicate the given prompt using the following method: 
            Please replace general concepts with more specific concepts.         
            You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #Given Prompt#.
            ‘#Given Prompt#’, ‘#Rewritten Prompt#’, ‘given prompt’ and ‘rewritten prompt’ are not allowed to appear in #Rewritten Prompt#.  
            #Given Prompt#:
            """ + instruction
            created_prompt = self.model(evol_prompt)
            state_encode = [0,0,0,1,0,0]
            cost = 1
            return created_prompt, state_encode, cost
        elif action == "increase_reasoning_steps":
            evol_prompt = """
            I want you act as a Prompt Rewriter.
            Your objective is to rewrite a given prompt into a more complex version to make those famous AI systems (e.g., chatgpt and GPT4) a bit harder to handle. 
            But the rewritten prompt must be reasonable and must be understood and responded by humans. 
            Your rewriting cannot omit the non-text parts such as the table and code in #Given Prompt#:. Also, please do not omit the input in #Given Prompt#.
            Don't repeat the conditions and requirements in the response, and Don't disclose your role.
            The Prompt Rewriter Must not give the introduction and explain the reason, the Prompt Rewriter must just give the most relevant response.
            This new prompt should not exceed 2048 words.
            You SHOULD complicate the given prompt using the following method: 
            If #Given Prompt# can be solved with just a few simple thinking processes, you can rewrite it to explicitly request multiple-step reasoning. 
            You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #Given Prompt#.
            ‘#Given Prompt#’, ‘#Rewritten Prompt#’, ‘given prompt’ and ‘rewritten prompt’ are not allowed to appear in #Rewritten Prompt#. 
            #Given Prompt#:
            """ + instruction
            created_prompt = self.model(evol_prompt)
            state_encode = [0,0,0,0,1,0]
            cost = 1
            return created_prompt, state_encode, cost
        elif action == "complicate_input":
            evol_prompt = """
            I want you act as a Prompt Rewriter. Your objective is to rewrite a given prompt into a more complex version using dataformat to make those famous AI systems (e.g., chatgpt and GPT4) more difficult to handle. 
            But the rewritten prompt must be reasonable and must be understood and responded by humans. 
            Don't repeat the conditions and requirements in the response, and Don't disclose your role.
            The Prompt Rewriter Must not give the introduction and explain the reason, the Prompt Rewriter must just give the most relevant response. 
            This new prompt should not exceed 2048 words.           
            The Given Prompt:  
            """+ instruction
            created_prompt = self.model(evol_prompt)
            state_encode = [0,0,0,0,0,1]
            cost = 1
            return created_prompt, state_encode, cost
        elif action == "none_query_action":
            # evol_prompt = """
            # I want you act as a Prompt Rewriter. Your objective is to rewrite a given prompt into a more complex version using dataformat to make those famous AI systems (e.g., chatgpt and GPT4) more difficult to handle. 
            # But the rewritten prompt must be reasonable and must be understood and responded by humans. 
            # Don't repeat the conditions and requirements in the response, and Don't disclose your role.
            # The Prompt Rewriter Must not give the introduction and explain the reason, the Prompt Rewriter must just give the most relevant response. 
            # This new prompt should not exceed 2048 words.           
            # The Given Prompt:  
            # """+ instruction
            # created_prompt = self.model(evol_prompt)
            state_encode = [0,0,0,0,0,0]
            cost = 0
            return instruction, state_encode, cost


    def step(self, action, instruction):
        judge_prompt = """
                1. The evolved instruction does not provide any information increment compared to the original 
                instruction. We use the following prompt to make this determination:
                Here are two Instructions to ChatGPT AI, do you think they are equal to each other, which meet the following requirements:
                1). They have same constraints and requirments.
                2). They have same depth and breadth of the inquiry.
                The First Prompt: <Here is first instruction.>
                The Second Prompt: <Here is second instruction.>
                Your Judgement (Just answer: Equal or Not Equal. No need to explain the reason.):
                2. The evolved instruction makes it difficult for the large language model to generate a response.
                We found that when the generated response contains “sorry” and is relatively short in length (i.e., less than 80 words), it often indicates that the large language model struggles to respond to the evolved instruction. So we can use this rule to make a judgment.
                3. The response generated by the large language model only contains punctuation and stop words.
                4. The evolved instruction obviously copies some words from the evolving prompt, such as “given prompt”, “rewritten prompt”, “#Rewritten Prompt#”, etc.
                """
        # print("action----test1:", action)
        # print("action----type:", type(action))
        action = torch.Tensor(action)
        # print("action----tensor:", action)
        if action.equal(torch.Tensor([1,0,0,0,0,0])):
            action = "breadth_action"
        elif action.equal(torch.Tensor([0,1,0,0,0,0])):
            action = "add_constraints"
        elif action.equal(torch.Tensor([0,0,1,0,0,0])):
            action = "deepening"
        elif action.equal(torch.Tensor([0,0,0,1,0,0])):
            action = "concretizing"
        elif action.equal(torch.Tensor([0,0,0,0,1,0])): 
            action ="increase_reasoning_steps"  
        elif action.equal(torch.Tensor([0,0,0,0,0,1])):
            action = "complicate_input"
        else:
            action = "breadth_action" #"none_query_action" 
            
        state, state_encode, cost_action = self.action_take(action, instruction)       

        # The evolved instruction does not provide any information increment compared to the original instruction. We use the following prompt to make this determination:
        judge_prompt_1 = """
        Here are two Instructions to ChatGPT AI, do you think they are equal to each other, which meet the following requirements:
        1. They have same constraints and requirments.
        2. They have same depth and breadth of the inquiry.
        The First Instruction: """ +instruction + """.
        The Second Instruction: """ + state + """.
        Your Judgement (Must Just only answer: Equal or Not Equal. No need to explain the reason; ‘Equal’ and ‘Not Equal’ are not allowed to appear simultaneously.):  
                    """
        reward_1 = 0
        cost_1 = 0        

        reward = reward_1 #+ reward_2 + reward_3 + reward_4
        cost = cost_1 # + cost_2 + cost_3 + cost_4
        
        

        file_path = folder + '/prompt_state.json'
        self.append_to_json_file(file_path=file_path, new_data=state)

        cost = cost_action
        done = False
        truncated = False
        info = {
            "reward": reward,
            "cost": -cost,
        }

        return state, state_encode, reward, done, truncated, info #state, state_encode, reward, cost, info









