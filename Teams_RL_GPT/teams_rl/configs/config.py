import argparse


def get_config():
    parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
    parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                        help='discount factor (default: 0.995)')
    parser.add_argument('--env-name', default="RL-Evol", metavar='G',
                        help='name of the environment to run')
    parser.add_argument('--algo', default="Teams_RL", metavar='G',
                        help='name of the algorithm to run')
    parser.add_argument('--slack-bound', type=float, default=0.005, metavar='G', # 0.04-0.045
                        help='slack bound (default: 0.02)')
    parser.add_argument('--exploration-iteration', type=int, default=1, metavar='G',
                        help='the epoch number of the first performance exploration (default: 40)')
    parser.add_argument('--exps-epoch', type=int, default=500, metavar='G',
                        help='the epoch number of exps (default: 500)')
    parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                        help='gae (default: 0.97)')
    parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                        help='l2 regularization regression (default: 1e-3)')
    parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                        help='max kl value (default: 1e-2)')
    parser.add_argument('--damping', type=float, default=1e-1, metavar='G',
                        help='damping (default: 1e-1)')
    parser.add_argument('--seed', type=int, default=543, metavar='N',
                        help='random seed (default: 1)')
    parser.add_argument('--batch-size', type=int, default=15000, metavar='N',
                        help='random seed (default: 1)')
    parser.add_argument('--render', action='store_true',
                        help='render the environment')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='interval between training status logs (default: 10)')
    return parser