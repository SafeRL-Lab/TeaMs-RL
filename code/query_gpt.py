import json
import argparse
from itertools import count
from copy import deepcopy
import time
import pandas as pd
import os
import openai

from gpt_collect import gpt_call
from datetime import datetime

currentDateAndTime = datetime.now()
start_run_date_and_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

folder = os.getcwd()[:-12] + 'gpt_data/' + 'gpt_4_1106/' + str(
    start_run_date_and_time) + '/'


if not os.path.exists(folder):
    os.makedirs(folder)

# read JSON file
with open(
        'your-data-file.json',
        'r') as file:
    collect_gpt_data = json.load(file)

Epoch = int(len(collect_gpt_data)) + 1


def read_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []
    return data


def write_json_file(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def append_to_json_file(file_path, new_data):
    data = read_json_file(file_path)
    data.append(new_data)
    write_json_file(file_path, data)


file_path = folder + '/instruction_response.json'

folder_ID_episode = os.getcwd()[:-12] + 'gpt_data/'
file_path_ID_episode = folder_ID_episode
filename_ID_episode = 'quert_gpt4_1106_ID_episode.json'
# check path
if not os.path.exists(file_path_ID_episode):
    os.makedirs(file_path_ID_episode)
# save JSON file
file_path_ID_episode = os.path.join(file_path_ID_episode, filename_ID_episode)
# chech file
if os.path.exists(file_path_ID_episode):
    # read data from files
    with open(file_path_ID_episode, 'r') as file:
        try:
            data = json.load(file)
            start_iteration = data["iteration_count"]

        except (json.JSONDecodeError, KeyError):
            start_iteration = 0
else:
    # if files do not exist, create a file 
    start_iteration = 0
    with open(file_path_ID_episode, 'w') as file:
        json.dump({"iteration_count": start_iteration}, file)

for i_episode in count(start_iteration):
    prompt = collect_gpt_data[i_episode]["instruction"] + "\n" + collect_gpt_data[i_episode]["input"]

    # prompt_state = gpt3_call(prompt)
    # prompt_state = prompt_state["choices"][0]["message"]["content"]
    # collect_gpt_data[i_episode]["output"] = prompt_state
    # append_to_json_file(file_path=file_path, new_data=collect_gpt_data[i_episode])

    try:
        prompt_state = gpt3_call(prompt)
        # prompt_state = prompt_state["choices"][0]["message"]["content"]
        collect_gpt_data[i_episode]["output"] = prompt_state
        append_to_json_file(file_path=file_path, new_data=collect_gpt_data[i_episode])
    except openai.error.InvalidRequestError:
        start_iteration = i_episode + 1
        print('\033[0;33m "Exception happens!" \033[0m', start_iteration)

    # model_evaluate = Call_model()
    # prompt_state = model_evaluate.evaluate(prompt)
    print("Success--------", str(i_episode) + " / " + str(Epoch))

    # write current iteration into JSON file
    with open(file_path_ID_episode, 'w') as file:
        json.dump({"iteration_count": i_episode + 1}, file)

    if i_episode > Epoch:
        break


