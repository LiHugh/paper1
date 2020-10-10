import logging
import pickle
import random
from influence import influence
import numpy as np
from change_baseline import Change
from net_env import NetworkEnv
from implementation_alg import PriortizedReplay
from dqn import DQNTrainer
from implementation_alg import OrnsteinUhlenbeckActionNoise
import networkx as nx
from deepwalk import DeepWalk
from multiprocessing import cpu_count
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import gc


budget = 5
nop_reward = 0
times_mean_env = 5
bad_reward = 0
max_reward = None
min_reward = None
norm_reward = 0
BUFF_SIZE = 4000
NUM_EP = 10
LR = 0.0001
input_dim = 20
action_dim = 60
const_features = 1
if const_features:
    input_dim = input_dim
else:
    input_dim = action_dim + 2
use_cuda = 1
gamma = 0.99
eta = 0.1
epsilon = 0.1
batch_size = 100
gcn_layers = 2
num_pooling = 1
assign_dim = 100
assign_hidden_dim = 150
noise_momentum = 0.15
noise_magnitude = 0.2
changeSeeds = 0


rg1 = np.random.RandomState(10)
num_walks = 80
walk_len = 10
cpu = 0
win = 5
emb_iters = 50
logdir = None
write = 1
save_every = 100
save_model = 'sample_'
noise_decay_rate = 0.999
eta_decay = 1
eps_decay_rate = 0.999
noise_param = 1


g_paths = [
    'data/rt/copen.pkl',
    # 'data/rt/assad.pkl',
    # 'data/rt/damascus.pkl',
    # 'data/rt/israel.pkl',
    # 'data/rt/obama.pkl',
    # 'data/rt/tlot.pkl',
    # 'data/rt/voteonedirection.pkl',
    # 'data/rt/occupy.pkl'
]

# ##########################################################################################################
syn = False
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
random.seed(10)
# ############################################################################################################


graphs = []
for g_path in g_paths:
    with open(g_path, 'rb') as fl:
        graphs.append(pickle.load(fl))
    print(g_path)
    g = graphs[-1]
    print("Nodes:", len(g))
    print("Edges:", len(g.edges))
    logging.info("Nodes: " + str(len(g)) + ' Edges: ' + str(len(g.edges)))

# Get best baseline of OPT
opts = []
for gp, g in zip(g_paths, graphs):
    opt_obj, local_obj, S_opt = influence(g, g)
    print(gp)
    print('OPT Results:', opt_obj, S_opt)
    logging.info('OPT Results:' + str(opt_obj) + '' + str(S_opt))

# initialize the seeds
rg = np.random.RandomState(10)
extra_seeds = 5
# Initialize seeds
e_seeds_list = []
for g in graphs:
    e_seeds_list.append(list(rg.choice(len(g), extra_seeds)))
# e_seeds = [31, 171]
print('Initialize Seeds:', e_seeds_list)

# Get the result of the CHANGE algorithm
ch = []
for gp, g in zip(g_paths, graphs):
    rs = []
    for _ in range(5):
        change = Change(g, budget=budget * 2, seeds=[])
        obj1, local_obj1, S1 = change()
        rs.append(obj1)
    ch.append(np.mean(rs))
    print("Change for %s is %f" % (gp, ch[-1]))
logging.info('Change Results:' + str(obj1) + ' ' + str(S1))


# min_goal = args.min_goal
# mid_goal = args.mid_goal
# max_goal = args.max_goal
goals = []
print('goal setting')
for gp, g, c in zip(g_paths, graphs, ch):
    opt_obj, local_obj, S_opt = influence(g, g)
    min_goal = int(c)
    mid_goal = int((c + opt_obj)/2)
    max_goal = int(opt_obj)
    print('Min_goal:', min_goal)
    print('Mid_goal:', mid_goal)
    print('Max_goal:', max_goal)
    goals.append([min_goal, mid_goal, max_goal])
    print('Goals is:', goals)
# logging.info('Goals setting is:' + str(goals))
print('goal setting over')


try:
    # import numpy as np
    # goals = [[24, 34, 44]]
    # rg1 = np.random.RandomState(10)
    # idx = rg1.randint(1)
    idx = rg1.randint(len(graphs))
    # print(idx)
    goal = goals[idx]
    # print(goal)
    min_goal = goal[0]
    mid_goal = goal[1]
    max_goal = goal[2]
    print('goal setting in try is:', max_goal)
except KeyboardInterrupt:
    print('Error!')



# Add the environment
envs = []
for g, seeds in zip(graphs, e_seeds_list):
    env = NetworkEnv(fullGraph=g, seeds=seeds, opt_reward=0, nop_r=nop_reward,
                     times_mean=times_mean_env, bad_reward=bad_reward, clip_max=max_reward,
                     clip_min=min_reward, normalize=norm_reward)
    envs.append(env)
# PriortizedReplay Buff
replay = PriortizedReplay(BUFF_SIZE, 10, beta=0.6)

# Model
acmodel = DQNTrainer(input_dim=input_dim, state_dim=action_dim, action_dim=action_dim, replayBuff=replay, lr=LR,
                     use_cuda=use_cuda, gamma=gamma,
                     eta=eta, gcn_num_layers=gcn_layers, num_pooling=num_pooling, assign_dim=assign_dim,
                     assign_hidden_dim=assign_hidden_dim)

noise = OrnsteinUhlenbeckActionNoise(action_dim, theta=noise_momentum, sigma=noise_magnitude)


# Embedding graph
def get_embeds(g):
    d = {}
    for n in g.nodes:
        d[n] = str(n)
    g1 = nx.relabel_nodes(g, d)
    graph_model = DeepWalk(g1, num_walks=num_walks, walk_length=walk_len,
                           workers=cpu if cpu > 0 else cpu_count())

    graph_model.train(window_size=win, iter=emb_iters, embed_size=action_dim)
    embs = {}
    emb1 = graph_model.get_embeddings()
    for n in emb1.keys():
        embs[int(n)] = emb1[n]

    return embs


# Get the N*input_dim matrix
def make_const_attrs(graph, input_dim):
    n = len(graph)
    mat = np.ones((n, input_dim))
    # mat = np.random.rand(n,input_dim)
    return mat


def make_env_attrs_1(env, embs, n=len(g), input_dim=input_dim):
    mat1 = np.zeros((n, int(action_dim + 2)))
    for u in env.active:
        mat1[u, :-2] = embs[u]
        mat1[u, -2] = 1
    for u in env.possible_actions:
        mat1[u, :-2] = embs[u]
        mat1[u, -1] = 1
    return mat1


def get_action_curr1(s, emb, nodes):
    q_vals = -10000.0
    node = -1
    for v in nodes:
        value, _ = acmodel.get_values2(s[0], s[1], emb[v])
        if value > q_vals:
            q_vals = value
            node = v
    return node, q_vals


def get_action_curr2(s, emb, nodes):
    q_vals = -10000.0
    node = -1
    for v in nodes:
        _, value = acmodel.get_values2(s[0], s[1], emb[v])
        if value > q_vals:
            q_vals = value
            node = v
    return node, q_vals


def get_action(s, emb, nodes):
    q_vals = -10000.0
    node = -1
    for v in nodes:
        value, _ = acmodel.get_values2_(s[0], s[1], emb[v])
        if value > q_vals:
            q_vals = value
            node = v
    return node, q_vals


node_attrs = make_const_attrs(g, input_dim)
n_iter = 0
rws = []


# if logdir is None:
#     writer = SummaryWriter()
# else:
#     writer = SummaryWriter(os.path.join('runs', logdir))


idx = 0
# print(len(graphs))


# env = envs[idx]
# g = graphs[idx]
# opt = opts[idx]
# change_score = ch[idx]
# print("Choosing %s" % (g_paths[idx]))]\

# print('OPT Result is %s' % opt)
# print('CHANGE Score is %s' % change_score)

# try:
#     for ep in range(NUM_EP):
#         # multiprocessing.freeze_support()
#         idx = rg1.randint(len(graphs))
#         env = envs[idx]
#         g = graphs[idx]
#         opt = opts[idx]
#         change_score = ch[idx]
#
#         print("Choosing %s" % (g_paths[idx]))
#
#         res = []
#         if changeSeeds:
#             e_seeds = list(rg.choice(len(g), extra_seeds))
#         else:
#             e_seeds = e_seeds_list[idx]
#
#         env.reset(seeds=e_seeds)
#         node_list = list(env.active.union(env.possible_actions))
#
#         t = env.state
#         t0 = len(t)
#         s_embs = get_embeds(env.sub)
#         if const_features:
#             s = [node_attrs[node_list], env.state]
#         else:
#             s = [make_env_attrs_1(env=env, embs=s_embs, n=len(g))[node_list], env.state]
#         print('Episode:', ep)
#         print('Seeds:', e_seeds)
#         tot_r = 0
#         tot_r1 = 0
#         for stps in range(budget):
#
#             possible_actions = [node_list.index(x) for x in env.possible_actions]
#
#             state_embed, _ = acmodel.get_node_embeddings(nodes_attr=s[0], adj=s[1], nodes=possible_actions)
#             l = list(env.possible_actions)
#             possible_actions_embed = [s_embs[x] for x in l]
#
#             if rg1.rand() > epsilon and (replay.size > batch_size or ep == 0):
#
#                 if rg1.rand() > 0.5:
#                     actual_action, q = get_action_curr1(s, s_embs, l)
#                 else:
#                     actual_action, q = get_action_curr2(s, s_embs, l)
#                 proto_action = actual_action_embed = s_embs[actual_action]
#
#
#             else:
#                 actual_action = rg1.choice(list(env.possible_actions), 1)[0]
#                 proto_action = actual_action_embed = s_embs[actual_action]
#
#             res.append(actual_action)
#
#             _, r, d, _ = env.step(actual_action)
#
#             node_list = list(env.active.union(env.possible_actions))
#             t = env.state
#             s_embs = get_embeds(env.sub)
#
#             if const_features:
#                 s1 = [node_attrs[node_list], env.state]
#             else:
#                 s1 = [make_env_attrs_1(env=env, embs=s_embs, n=len(g))[node_list], env.state]
#
#             logging.debug('State: ' + str(state_embed))
#             logging.debug('Action:' + str(proto_action))
#
#             # if last time step or explored entire graph
#             if stps == budget - 1 or len(env.possible_actions) == 0:
#                 env.step(-1)
#                 r += env.reward
#                 d = True
#             if d:
#                 s1[1] *= 0
#             tot_r += r
#
#             t = len(env.state)
#             # sub = nx.from_numpy_matrix(t)
#             # b,_,_ = influence(sub, sub)
#             r1 = r + (1 / (len(g))) * (t - t0)
#             t0 = t
#             if d:
#                 r1 = r1 / opt
#
#             tot_r1 += r1
#
#             # TODO: TD Compute
#             td = acmodel.td_compute(s, actual_action_embed, r1, s1,
#                                     s_embs[get_action(s1, s_embs, env.possible_actions)[0]])
#             replay.add(s, actual_action_embed, r1, s1, s_embs[get_action(s1, s_embs, env.possible_actions)[0]],
#                        actual_action, td=np.abs(td))
#
#             if (ep == 0 and stps < 2) or replay.size > batch_size:
#                 acmodel.gradient_update_sarsa(batch_size=batch_size)
#                 acmodel.gradient_update_sarsa(batch_size=batch_size)
#             torch.cuda.empty_cache()
#
#             n_iter += 1
#             if write:
#                 writer.add_scalar('CriticLoss', acmodel.loss_critic.clone().cpu().data.numpy(), n_iter)
#
#             s = s1
#             if d:
#                 break
#
#         print('Critic Loss:', acmodel.loss_critic)
#         print('Action:', proto_action)
#         print('Value:', q)
#         print('Env Reward:', r1)
#         print('Reward:', tot_r)
#         print('Chosen:', res, '\n')
#         logging.info('Episode: ' + str(ep) + ' Reward: ' + str(tot_r))
#         logging.debug('Critic Loss: ' + str(acmodel.loss_critic))
#         rws.append(tot_r)
#
#         if write:
#             writer.add_scalar('Reward', tot_r, ep + 1)
#             writer.add_scalar('Influence', env.reward_, ep + 1)
#             writer.add_scalar('Norm Reward', tot_r1, ep + 1)
#
#         gc.collect()
#
#         if ep % save_every == 0:
#             # acmodel.save_models(args.save_model)
#             torch.save(acmodel, 'models/' + save_model + str(ep) + '.pth')
#
#         noise_param *= max(0.001, noise_decay_rate)
#         acmodel.eta = max(0.001, acmodel.eta * eta_decay)
#         epsilon = max(0.01, epsilon * eps_decay_rate)
#
#     writer.close()
#
# except KeyboardInterrupt:
#     writer.close()
#
#
