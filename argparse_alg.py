# 1：import argparse
#
# 2：parser = argparse.ArgumentParser()
#
# 3：parser.add_argument()
#
# 4：parser.parse_args()
# parser = argparse.ArgumentParser(description='Process some integers.')
# parser.add_argument('integers', metavar='N', type=int, nargs='+',
#                     help='an integer for the accumulator')
# parser.add_argument('--sum', dest='accumulate', action='store_const',
#                     const=sum, default=max,
#                     help='sum the integers (default: find the max)')
#
# args = parser.parse_args()
# print(args.accumulate(args.integers))

# ArgumentParser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
# 定义单个的命令行参数应当如何解析。每个形参都在下面有它自己更多的描述，长话短说有：
#
# name or flags - 一个命名或者一个选项字符串的列表，例如 foo 或 -f, --foo。
#
# action - 当参数在命令行中出现时使用的动作基本类型。
#
# nargs - 命令行参数应当消耗的数目。
#
# const - 被一些 action 和 nargs 选择所需求的常数。
#
# default - 当参数未在命令行中出现时使用的值。
#
# type - 命令行参数应当被转换成的类型。
#
# choices - 可用的参数的容器。
#
# required - 此命令行选项是否可省略 （仅选项可用）。
#
# help - 一个此选项作用的简单描述。
#
# metavar - 在使用方法消息中使用的参数值示例。
#
# dest - 被添加到 parse_args() 所返回对象上的属性名。

import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='Influence Maxima Arguments')
    parser.add_argument('--logfile', dest='logfile', type=str, default='train.log',
                        help='Logging file')
    parser.add_argument('--logdir', dest='logdir', type=str, default=None,
                        help='Tensorboard LogDir')
    parser.add_argument('--log-level', dest='loglevel', type=int, default=2, choices=[1, 2],
                        help='Logging level')
    parser.add_argument('--sample-budget', dest='budget', type=int, default=5,
                        help='Number of queries for sampling')
    parser.add_argument('--extra-seeds', dest='extra_seeds', type=int, default=5,
                        help='Initial number of random seeds')
    parser.add_argument('--prop-prob', dest='prop_probab', type=float, default=0.1,
                        help='Propogation Probability for each Node')
    parser.add_argument('--cpu', dest='cpu', type=int, default=0,
                        help='Number of CPUs to use for influence sampling')
    parser.add_argument('--samples', dest='samples', type=int, default=100,
                        help='Number of samples in Influence Maximization')
    parser.add_argument('--opt', dest='obj', type=float, default=0,
                        help='Threshold for reward')
    parser.add_argument('--infl-budget', dest='ibudget', type=int, default=10,
                        help='Number of queries during influence(greedy steps)')
    parser.add_argument('--render', dest='render', type=int, default=0,
                        help='1 to Render graphs, 0 to not')
    parser.add_argument('--write', dest='write', type=int, default=1,
                        help='1 to write stats to tensorboard, 0 to not')

    parser.add_argument('--change-seeds', dest='changeSeeds', type=int, default=0,
                        help='1 to change seeds after each episode, 0 to not')
    parser.add_argument('--add-noise', dest='add_noise', type=int, default=1,
                        help='1 to add noise to action 0 to not')
    parser.add_argument('--sep-net', dest='sep_net', type=int, default=0,
                        help='Seperate network rep for actor and critic')

    parser.add_argument('--save-freq', dest='save_every', type=int, default=100,
                        help='Model save frequency')

    parser.add_argument('--eps', dest='num_ep', type=int, default=10000,
                        help='Number of Episodes')
    parser.add_argument('--buffer-size', dest='buff_size', type=int, default=4000,
                        help='Replay buffer Size')

    parser.add_argument('--gcn_layers', dest='gcn_layers', type=int, default=2,
                        help='No. of GN Layers before each pooling')
    parser.add_argument('--num_poolig', dest='num_pooling', type=int, default=1,
                        help='No.pooling layers')
    parser.add_argument('--assign_dim', dest='assign_dim', type=int, default=100,
                        help='pooling hidden dims 1')
    parser.add_argument('--assign_hidden_dim', dest='assign_hidden_dim', type=int, default=150,
                        help='pooling hidden dims 2')

    parser.add_argument('--actiondim', dest='action_dim', type=int, default=60,
                        help='Action(Node) Dimensions')
    parser.add_argument('--const_features', dest='const_features', type=int, default=1,
                        help='1 to have constant features')
    parser.add_argument('--inputdim', dest='input_dim', type=int, default=20,
                        help='Node features Dimensions')

    parser.add_argument('--step_reward', dest='nop_reward', type=float, default=0,
                        help='Reward for each step')
    parser.add_argument('--bad_reward', dest='bad_reward', type=float, default=0,
                        help='Reward for each step that is closer to active')
    parser.add_argument('--norm_reward', dest='norm_reward', type=int, default=0,
                        help='Normalize reward with opt')
    parser.add_argument('--max_reward', dest='max_reward', type=int, default=None,
                        help='Normalize reward with opt')
    parser.add_argument('--min_reward', dest='min_reward', type=int, default=None,
                        help='Normalize reward with opt')

    parser.add_argument('--lr', dest='lr', type=float, default=1e-4,
                        help='Learning Rate')
    parser.add_argument('--eta', dest='eta', type=float, default=0.1,
                        help='Target network transfer rate')
    parser.add_argument('--gamma', dest='gamma', type=float, default=0.99,
                        help='Discount rate')
    parser.add_argument('--epsilon', dest='epsilon', type=float, default=0.1,
                        help='Epsilon exploration')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=100,
                        help='Gradient Update Batch Size')

    parser.add_argument('--use_cuda', dest='use_cuda', type=int, default=1,
                        help='1 to use cuda 0 to not')
    parser.add_argument('--walk_len', dest='walk_len', type=int, default=10,
                        help='Walk Length')

    parser.add_argument('--num_walks', dest='num_walks', type=int, default=80,
                        help='Walk Length')
    parser.add_argument('--win', dest='win', type=int, default=5,
                        help='Window size')
    parser.add_argument('--emb_iters', dest='emb_iters', type=int, default=50,
                        help='Walk Length')

    parser.add_argument('--noise_momentum', dest='noise_momentum', type=float, default=0.15,
                        help='Noise Momentum')
    parser.add_argument('--noise_magnitude', dest='noise_magnitude', type=float, default=0.2,
                        help='Noise Magnitude')

    parser.add_argument('--noise_decay', dest='noise_decay_rate', type=float, default=0.999,
                        help='Noise Decay Rate')
    parser.add_argument('--eta_decay', dest='eta_decay', type=float, default=1.,
                        help='eta Decay Rate')
    parser.add_argument('--alpha_decay', dest='alpha_decay', type=float, default=1.,
                        help='alpha Decay Rate')
    parser.add_argument('--eps_decay', dest='eps_decay_rate', type=float, default=0.999,
                        help='Epsilon Decay Rate')

    parser.add_argument('--sample_times', dest='times_mean', type=int, default=10,
                        help='Number of times to sample objective from fluence algorithm')
    parser.add_argument('--sample_times_env', dest='times_mean_env', type=int, default=5,
                        help='Number of times to sample objective from fluence algorithm for env rewards')
    parser.add_argument('--save_model', dest='save_model', type=str, default='sample_',
                        help='Name of Save model')
    parser.add_argument('--neigh', dest='k', type=int, default=1,
                        help='K nearest for ation')

    return parser.parse_args()