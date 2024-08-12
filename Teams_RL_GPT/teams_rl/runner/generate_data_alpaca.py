

import argparse
from itertools import count
from copy import deepcopy
import time
import gym
import scipy.optimize
import random

from torch.autograd import Variable
import torch
import json

import pandas as pd

from teams_rl.models.models import *
from teams_rl.algorithms.trpo import trpo_step, pcgrad
from teams_rl.utils.replay_memory import *
from teams_rl.utils.replay_memory import Memory
from teams_rl.utils.running_state import ZFilter
from teams_rl.utils.utils import *

from teams_rl.configs.config import get_config
from teams_rl.configs.config_mujoco import get_env_mujoco_config
import os
import time
from datetime import datetime
currentDateAndTime = datetime.now()
start_run_date_and_time=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())




torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')
args = get_config().parse_args()
env = get_env_mujoco_config(args)


folder = os.getcwd()[:-6] + 'data_VT/' + 'gpt-4-1106-preview_40k_41k_prompt_Alpaca'+args.algo+'-'+ args.env_name + '/seed-' + str(args.seed) + '-' + str(
        start_run_date_and_time) + '/'

folder_ID_episode = os.getcwd()[:-6] + 'data_VT' 
# folder = os.getcwd()[:-4] + 'runs\\test\\'
if not os.path.exists(folder):
    os.makedirs(folder)

argsDict = args.__dict__
with open(folder + '/config.json', 'w') as f:
    f.writelines('------------------ start ------------------' + '\n')
    for eachArg, value in argsDict.items():
        f.writelines(eachArg + ' : ' + str(value) + '\n')
    f.writelines('------------------- end -------------------')

with open(folder + '/Generate_Alpaca.json', 'w') as f:
    f.writelines('------------------ start ------------------' + '\n')


# env = gym.make(args.env_name)
# env = HalfCheetahEnv(goal_vel=0.6)
# env = Walker2dEnv()
num_inputs = len(env.observation_space)
num_actions = len(env.action_space)

# env.seed(args.seed)
torch.manual_seed(args.seed)

state_dict = torch.load('/home/your_policy_model_path/model128.pth')


policy_net = Policy(num_inputs, num_actions)
policy_net.load_state_dict(state_dict) 
# print("policy net------------:", policy_net)
value_net = Value(num_inputs)


def select_action(state):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    # print("action_mean------:", action_mean)
    action = torch.normal(action_mean, action_std)
    action[0][1] = action[0][1] + 0.7 #1 #1.5
    # print("action--value----------:", action)
    # max_index = action.index(max(action[0]))
    # print("state----:", state)
    max_index = torch.argmax(action[0])
    init_action = torch.FloatTensor(1, len(state[0])).fill_(0) #[[0]*len(state)]
    # print("init_max----------1:", init_action)
    init_action[0][max_index] = 1
    action = init_action
    # print("init_max----------2:", init_action)
    return action

  

running_state = ZFilter((num_inputs,), clip=5)
running_reward = ZFilter((1,), demean=False, clip=10)

# EPISODE_LENGTH = 1000
EPISODE_LENGTH = 6 # 7 # 30
# EPISODE_PER_BATCH = 16
EPISODE_PER_BATCH = 1
# Epoch = 500
Epoch = 200
alpha = 0.02
prev_policy_net = deepcopy(policy_net)


# init_instruction = "How to write an academic paper?"
# init_instruction = "How to cook food?"

with open('/home/gushangding/your_data_path/data/alpaca_data.json', 'r') as file:  
    alpaca_data = json.load(file)  
  
# alpaca_data[0]["instruction"]

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

state_datas = []
velocity_datas = []
pos_datas = []
reward_datas = []
average_step_reward_datas = []
goal_vels = []
goal_vels.append(0.6)

file_path_ID_episode = folder_ID_episode
filename_ID_episode = 'ID_episode.json'
# check path
if not os.path.exists(file_path_ID_episode):  
    os.makedirs(file_path_ID_episode)  
  
# save json to path
file_path_ID_episode = os.path.join(file_path_ID_episode, filename_ID_episode)  


# check file
if os.path.exists(file_path_ID_episode):  
    # if files exit, read the file number  
    with open(file_path_ID_episode, 'r') as file:  
        try:  
            data = json.load(file)  
            start_iteration = data["iteration_count"]  
        except (json.JSONDecodeError, KeyError):  
            start_iteration = 40000 #15000
else:  
    #
    start_iteration = 40000  
    with open(file_path_ID_episode, 'w') as file:  
        json.dump({"iteration_count": start_iteration}, file) 



for i_episode in count(start_iteration): # 455
    memory = Memory()
    num_steps = 0
    reward_batch = 0
    num_episodes = 0
    state = 0
    tic = time.perf_counter()
    state_data = []
    velocity_data = []
    pos_data = []
    reward_data = []
    while num_steps < EPISODE_LENGTH*EPISODE_PER_BATCH:
        state, info = env.reset()
        state = running_state(state)
        reward_sum = 0
        for t in range(EPISODE_LENGTH): # Don't infinite loop while learning
            action = select_action(state)
            action = action.data[0].numpy()
            if t == 1:
                action = [1., 0., 0., 0., 0., 0.]
            elif t == 4:
                action = [1., 0., 0., 0., 0., 0.]
            elif np.array_equal(action, [1., 0., 0., 0., 0., 0.]):
                action = [0., 1., 0., 0., 0., 0.]
            prompt_state, next_state, reward, done, truncated, info = env.step(action, alpaca_data[i_episode]["instruction"])
            init_instruction = prompt_state
            
            reward_sum += info["reward"]
            state_data.append(state)
            reward_data.append(info["reward"])

            file_path = folder + '/Generate_Alpaca.json'
            alpaca_data[i_episode]["instruction"] = prompt_state
            append_to_json_file(file_path=file_path, new_data=alpaca_data[i_episode])            

            next_state = running_state(next_state)

            mask = 1
            if t==EPISODE_LENGTH-1:
                mask = 0

            memory.push(state, np.array([action]), mask, next_state, reward)
            # if args.render:
            #     env.render()
            if done or truncated:
                break

            state = next_state
        num_steps += EPISODE_LENGTH
        num_episodes += 1
        reward_batch += reward_sum
    state_datas.append(state_data)
    reward_datas.append(reward_data)
    average_step_reward_datas.append(reward_batch / num_steps)
    batch = memory.sample()
    iterations = 5   
    with open(file_path_ID_episode, 'w') as file:  
        json.dump({"iteration_count": i_episode + 1}, file)  


    if i_episode % 100 == 0: 
        model_file_path = os.path.join(folder, 'model'+str(i_episode)+'.pth')  
        torch.save(policy_net.state_dict(), model_file_path)    

    np.save(folder + "/average_step_reward.npy", np.array(average_step_reward_datas))

    npfile = np.load(folder + "/average_step_reward.npy")
    np_to_csv = pd.DataFrame(data=npfile)
    np_to_csv.to_csv(folder + "/average_step_reward"+ '.csv')
    np_to_csv = pd.DataFrame(data=npfile)

    if i_episode % args.log_interval == 0:
        print(f'Episode {i_episode}\tAverage step reward {reward_batch/num_steps:.2f}\t')
    if i_episode >= args.exps_epoch:
        break



