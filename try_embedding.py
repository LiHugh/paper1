from deepwalk import DeepWalk
import networkx as nx
import pickle
import numpy as np
from numpy.linalg import norm

from practice_2nd_net_env import NetworkEnv

from multiprocessing import cpu_count
import matplotlib.pyplot as plt
from dqn import DQNTrainer


# walk length
num_walks = 50
walk_len = 10
cpu = 0
win = 5
emb_iters = 30
action_dim = 10

g_paths = [
    'data/rt/copen.pkl',
    # 'data/rt/occupy.pkl'
]
# generate graph
graphs = []
for g_path in g_paths:
    with open(g_path, 'rb') as fl:
        graphs.append(pickle.load(fl))
    print(g_path)
    g = graphs[-1]
    print("Nodes:", len(g))
    print("Edges:", len(g.edges))

rg = np.random.RandomState(10)
rg1 = np.random.RandomState(10)

e_seeds_list = []
extra_seeds = 500
for g in graphs:
    e_seeds_list.append(list(rg.choice(len(g), extra_seeds)))
    print(e_seeds_list)


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


envs = []
times_mean_env = 5
bad_reward = 0
max_reward = 0
min_reward = 0
norm_reward = 0

for g, seeds in zip(graphs, e_seeds_list):
    env = NetworkEnv(fullGraph=g, seeds=seeds, opt_reward=0, nop_r=0,
                     times_mean=times_mean_env, bad_reward=bad_reward, clip_max=max_reward,
                     clip_min=min_reward, normalize=norm_reward)

    envs.append(env)

idx = rg1.randint(len(graphs))
env = envs[idx]
g = graphs[idx]
e_seeds = e_seeds_list[idx]
print(e_seeds[4])
e_seeds2 = e_seeds
e_seeds2[4] = 368


t = env.state
print('initial seed:', e_seeds)
s_embs1 = get_embeds(env.sub)
print(s_embs1)
print('############################3')
print('length for subgraph1', len(env.sub))
nx.draw(env.sub, with_labels=True, font_weight='bold')
plt.show()




env.reset(seeds=e_seeds2)
t = env.state
# nx.draw(env.sub)




s_embs2 = get_embeds(env.sub)
env.reset(seeds=e_seeds2)
print('initial seed:', e_seeds2)
print('length for subgraph2', len(env.sub))
print(s_embs2)
# nx.draw(env.sub, with_labels=True, font_weight='bold')
# plt.show()

for i,j in zip(s_embs2,s_embs1):
    print('node:', i, j)
    print(np.dot(s_embs2[i],s_embs1[j])/(norm(s_embs2[i])*norm(s_embs1[j])))
# print(np.dot(s_embs1[320],s_embs2[320]))
for i in s_embs1:
    print('same nodes in two similar graphs:', i, i)
    if i in s_embs2:
        print(np.dot(s_embs2[i],s_embs1[i])/(norm(s_embs2[i])*norm(s_embs1[i])))