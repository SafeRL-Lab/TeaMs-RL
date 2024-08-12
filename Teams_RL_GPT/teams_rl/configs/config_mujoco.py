from teams_rl.envs.rl_evol import Evol_gpt 
# from teams_rl.envs.rl_train_policy import Evol_gpt_policy

def get_env_mujoco_config(args):
    # env = gym.make(args.env_name)
    # todo: create new environments
    if args.env_name == "RL-Evol": # todo: need revise
        env = Evol_gpt()
        print('\033[0;35m "Create RL-Evol Environments!" \033[0m')
    elif args.env_name == "RL-Evol-Policy": # todo: need revise
        env = Evol_gpt_policy()
        print('\033[0;35m "Create RL-Evol Environments!" \033[0m')
    else:
        print('\033[0;31m "error! Please input a correct task name!" \033[0m')
    return env
