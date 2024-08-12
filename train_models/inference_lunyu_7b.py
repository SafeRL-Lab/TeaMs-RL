import sys
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import fire
import torch
# from peft import PeftModel
import transformers
# import gradio as gr
import json

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # （保证程序cuda序号与实际cuda序号对应）
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"  # （代表仅使用第0，1号GPU）

assert (
        "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass


base_model = "/home/your_model_path"


# assert base_model, (
#         "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
#     )

tokenizer = LlamaTokenizer.from_pretrained(base_model)
load_8bit = False
if device == "cuda":
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )
elif device == "mps":
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        device_map={"": device},
        torch_dtype=torch.float16,
    )

# unwind broken decapoda-research config
model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2

if not load_8bit:
    model.half()  # seems to fix bugs for some users.
if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)


# load_8bit: bool = False,
# input_data_path = "/path/to/WizardLM_testset.jsonl",
# output_data_path = "/path/to/WizardLM_testset_output.jsonl",
class Call_model():
    model.eval()

    def evaluate(self, instruction):
        final_output = self.inference(instruction + "\n\n### Response:")
        return final_output

    def inference(self,
                  batch_data,
                  input=None,
                  temperature=1,
                  top_p=0.95,
                  top_k=40,
                  num_beams=1,
                  max_new_tokens=4096,
                  **kwargs,
                  ):
        prompts = f"""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {batch_data} ASSISTANT:"""
        # prompts = batch_data
        inputs = tokenizer(prompts, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences
        output = tokenizer.batch_decode(s, skip_special_tokens=True)
        output = output[0].split("ASSISTANT:")[1].strip()
        return output

    


if __name__ == "__main__":
    # fire.Fire(main)
    # prompt = "What are the names of some famous actors that started their careers on Broadway?" #"How are you?"
    # prompt = input("Please input:")
    prompt = "Given a set of shoe size, add up the total size: Size 4, Size 7, Size 9"
    # "Step by step, how would you solve this equation? 3x + 6 = 24"
    # "Suppose I have 12 eggs. I drop 2 and eat 5. How many eggs do I have left?"
    # "Step by step, how would you solve this equation? (7x + 7) + (3x + 4) = 15"
    # "Find the 13th root of 1000"
    #"Identify all prime numbers between 50 and 60."
    # "What is the square root of 5929?"
    # "Identify three prime numbers between 1 and 10."
    # "Elaborate on the sequential methodology you would employ to isolate the variable within this intricate second-degree polynomial equation."
    # "Given a set of shoe size, add up the total size: Size 4, Size 7, Size 9"
    # "Given that f(x) = 5x^3 - 2x + 3, find the value of f(2)"
    #"Step by step, how would you solve this equation? (7x + 7)/(3x + 4) = 5" #"IDescribe a task that takes place at a dinner table."  # "Write a simple guide for uploading the bibliography database on Overleaf."
    prompt = str(prompt)
    model_evaluate = Call_model()
    prompt_state = model_evaluate.evaluate(prompt)
    # print("Output--------------:", prompt_state)
    print(prompt_state)
