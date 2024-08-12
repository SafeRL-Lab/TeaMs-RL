import argparse
from itertools import count
from copy import deepcopy
import time
import gym
import scipy.optimize
import random

from torch.autograd import Variable
import torch

from teams_rl.models.models import *
from teams_rl.algorithms.trpo import trpo_step, pcgrad, pcgrad_v1
from teams_rl.utils.replay_memory import *
from teams_rl.utils.replay_memory import Memory
from teams_rl.utils.running_state import ZFilter
from teams_rl.utils.utils import *

from teams_rl.configs.config import get_config
from teams_rl.configs.config_mujoco import get_env_mujoco_config

import pandas as pd

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

folder = os.getcwd()[:-6] + 'data/' + '2SR_policy-'+args.algo+'-'+ args.env_name + '/seed-' + str(args.seed) + '-' + str(
        start_run_date_and_time) + '/'
if not os.path.exists(folder):
    os.makedirs(folder)

argsDict = args.__dict__
with open(os.getcwd()[:-6] + 'data/' + '2SR_policy-'+ args.algo+'-'+ args.env_name + '/seed-' + str(args.seed) + '-' + str(
        start_run_date_and_time) + '/config.json', 'w') as f:
    f.writelines('------------------ start ------------------' + '\n')
    for eachArg, value in argsDict.items():
        f.writelines(eachArg + ' : ' + str(value) + '\n')
    f.writelines('------------------- end -------------------')


num_inputs = len(env.observation_space)
num_actions = len(env.action_space)

# env.seed(args.seed)
torch.manual_seed(args.seed)

policy_net = Policy(num_inputs, num_actions)
value_net = Value(num_inputs)

class Optimize_reward():
    def __init__(self, policy_net, get_value_loss, value_net, states, args, actions, advantages):
        self.get_value_loss  = get_value_loss
        self.states = states
        self.args = args
        self.policy_net = policy_net
        self.actions = actions
        self.advantages = advantages

        flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(self.get_value_loss,
                                                                get_flat_params_from(value_net).double().numpy(),
                                                                maxiter=25)
        set_flat_params_to(value_net, torch.Tensor(flat_params))
        self.advantages = (self.advantages - self.advantages.mean()) / self.advantages.std()
        action_means, action_log_stds, action_stds = self.policy_net(Variable(self.states))
        self.fixed_log_prob = normal_log_density(Variable(self.actions), action_means, action_log_stds, action_stds).data.clone()

    def get_loss_up(self, volatile=False):
        if volatile:
            with torch.no_grad():
                action_means, action_log_stds, action_stds = self.policy_net(Variable(self.states))
        else:
            action_means, action_log_stds, action_stds = self.policy_net(Variable(self.states))
        log_prob = normal_log_density(Variable(self.actions), action_means, action_log_stds, action_stds)
        action_loss = -Variable(self.advantages) * torch.exp(log_prob - Variable(self.fixed_log_prob))
        return action_loss.mean()

    def get_kl(self):
        mean1, log_std1, std1 = self.policy_net(Variable(self.states))
        mean0 = Variable(mean1.data)
        log_std0 = Variable(log_std1.data)
        std0 = Variable(std1.data)
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)
    def conduct_reward_trpo_step(self):
        trpo_step(self.policy_net, self.get_loss_up, self.get_kl, self.args.max_kl, self.args.damping)

def select_action(state):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    action = torch.normal(action_mean, action_std)
    action[0][0] = action[0][0] + 0.3
    max_index = torch.argmax(action[0])
    init_action = torch.FloatTensor(1, len(state[0])).fill_(0) 
    init_action[0][max_index] = 1
    action = init_action
    return action

def update_params(batch, i_episode):
    rewards = torch.Tensor(batch.reward)
    masks = torch.Tensor(batch.mask)
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    states = torch.Tensor(batch.state)
    values = value_net(Variable(states))

    returns = torch.Tensor(actions.size(0),1)
    deltas = torch.Tensor(actions.size(0),1)
    advantages = torch.Tensor(actions.size(0),1)    

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + args.gamma * args.tau * prev_advantage * masks[i]

        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]    

    targets = Variable(returns) 

    # Original code uses the same LBFGS to optimize the value loss
    def get_value_loss(flat_params):
        set_flat_params_to(value_net, torch.Tensor(flat_params))
        for param in value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)

        values_ = value_net(Variable(states))

        value_loss = (values_ - targets).pow(2).mean()

        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * args.l2_reg
        value_loss.backward()
        return (value_loss.data.double().numpy(), get_flat_grad_from(value_net).data.double().numpy())

    optimize_reward = Optimize_reward(policy_net, get_value_loss, value_net, states, args, actions, advantages)
    optimize_reward.conduct_reward_trpo_step()


def torch_save(self) -> None:
        """Save the torch model."""
        if self._maste_proc:
            assert self._what_to_save is not None, 'Please setup torch saver first'
            path = os.path.join(self._log_dir, 'torch_save', f'epoch-{self._epoch}.pt')
            os.makedirs(os.path.dirname(path), exist_ok=True)

            params = {
                k: v.state_dict() if hasattr(v, 'state_dict') else v
                for k, v in self._what_to_save.items()
            }
            torch.save(params, path)


running_state = ZFilter((num_inputs,), clip=5)
running_reward = ZFilter((1,), demean=False, clip=10)

# EPISODE_LENGTH = 1000
EPISODE_LENGTH = 6
# EPISODE_PER_BATCH = 16
EPISODE_PER_BATCH = 1
# Epoch = args.exps_epoch # 500
args.exps_epoch=200
alpha = 0.02
prev_policy_net = deepcopy(policy_net)
# prev_value_net = deepcopy(value_net)


# init_instruction = "How to write an academic paper?"
init_instruction = "How to cook food?"

state_datas = []
velocity_datas = []
pos_datas = []
reward_datas = []
average_step_reward_datas = []
goal_vels = []
goal_vels.append(0.6)
for i_episode in count(1):
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
            prompt_state, next_state, reward, done, truncated, info = env.step(action, init_instruction)
            init_instruction = prompt_state
            reward_sum += info["reward"]
            state_data.append(state)
            reward_data.append(info["reward"])

            next_state = running_state(next_state)

            mask = 1
            if t==EPISODE_LENGTH-1:
                mask = 0

            memory.push(state, np.array([action]), mask, next_state, reward)
            if done or truncated:
                break

            state = next_state
        num_steps += EPISODE_LENGTH
        num_episodes += 1
        reward_batch += reward_sum
    state_datas.append(state_data)
    reward_datas.append(reward_data)
    average_step_reward_datas.append(reward_batch/num_steps)
    batch = memory.sample()
    update_params(batch, i_episode)
    
    model_file_path = os.path.join(folder, 'model'+str(i_episode)+'.pth')  
    torch.save(policy_net.state_dict(), model_file_path)  
    np.save(folder + "/average_step_reward.npy", np.array(average_step_reward_datas))

    npfile = np.load(folder + "/average_step_reward.npy")
    np_to_csv = pd.DataFrame(data=npfile)
    np_to_csv.to_csv(folder + "/average_step_reward"+ '.csv')
    np_to_csv = pd.DataFrame(data=npfile)
    if i_episode % args.log_interval == 0:
        print(f'Episode {i_episode}\tAverage step reward {reward_batch/num_steps:.4f}\t')

    if i_episode >= args.exps_epoch:
        break
