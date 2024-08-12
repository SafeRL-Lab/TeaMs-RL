import sys
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import fire
import torch
# from peft import PeftModel
import transformers
# import gradio as gr
import json

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


base_model = "/your_model_path" # "/path/to/WizardLM13B",
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


class Call_model():
    model.eval()
    def evaluate(self, instruction):
        # print("instruction---------:", instruction)
        # print("X------------:", X)       
        # {instruction}\n\n### Response:
        final_output = self.inference(instruction+"\n\n### Response:")
        # print("final_output-----------------:", final_output)
        return final_output
    


    def inference(self, 
            batch_data,
            input=None,
            temperature=1,
            top_p=0.95,
            top_k=40,
            num_beams=1,
            max_new_tokens=2048,
            **kwargs,
    ):
        
        
        prompts = f"""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {batch_data} ASSISTANT:"""
        
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



    