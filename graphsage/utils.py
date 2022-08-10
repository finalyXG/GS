from __future__ import print_function

import numpy as np
import random
import json
import sys
import os

import networkx as nx
from networkx.readwrite import json_graph
version_info = list(map(int, nx.__version__.split('.')))
major = version_info[0]
minor = version_info[1]
# assert (major <= 1) and (minor <= 11), "networkx major version > 1.11"

WALK_LEN=5
N_WALKS=50

def load_data(prefix, normalize=True, load_walks=False, remove_isolated_nodes=False):
    G_data = json.load(open(prefix + "-G.json"))
    G = json_graph.node_link_graph(G_data)
    if isinstance(next(iter(G.nodes())), int):
        conversion = lambda n : int(n)
    else:
        conversion = lambda n : n

    # Laurence 20220705
    # Update can_graph attribute
    for n in G.nodes:
        G.nodes[n]['can_graph'] = G.degree(n) > 0

    if os.path.exists(prefix + "-feats.npy"):
        feats = np.load(prefix + "-feats.npy")
    elif 'feat' in G.nodes[next(iter(G.nodes))].keys():
        feats = np.array([G.nodes[n]['feat'] for n in iter(G.nodes)])
    else:
        print("No features present.. Only identity features will be used.")
        feats = None
    # Laurence 20220705
    if os.path.exists(prefix + "-id_map.json") == 0:
        id_map = {v1: v2 for v1,v2 in zip(list(G.nodes), range(len(G.nodes)))}
        # id_map = {v1: v2 for v1,v2 in zip(list(G.nodes), range(len(G.nodes)))}
        # id_map = {v1: v1 for v1 in list(G.nodes)}
    else:
        id_map = json.load(open(prefix + "-id_map.json"))
        id_map = {conversion(k):int(v) for k,v in id_map.items()}
    

    walks = []
    if os.path.exists((prefix + "-class_map.json")):
        class_map = json.load(open(prefix + "-class_map.json"))
    elif 'label' in G.nodes[next(iter(G.nodes))].keys():
        class_map = {e: G.nodes[e]['label'] for e in iter(G.nodes)}
    if isinstance(list(class_map.values())[0], list):
        lab_conversion = lambda n : n
    else:
        lab_conversion = lambda n : int(n)

    class_map = {conversion(k):lab_conversion(v) for k,v in class_map.items()}

    ## Remove all nodes that has zero degree if needed
    if remove_isolated_nodes:
        n_isolated_ls = [n for n in G.nodes if G.degree(n)==0]
        G.remove_nodes_from(n_isolated_ls)

    ## Remove all nodes that do not have val/test annotations
    ## (necessary because of networkx weirdness with the Reddit data)
    broken_count = 0
    for node in G.nodes():
        if not 'val' in G.nodes[node] or not 'test' in G.nodes[node]:
            G.remove_node(node)
            broken_count += 1
    print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))


    ## If no "val" exists, take a random 10 percentange of "train" data as "val"
    val_nb = sum([1 if (G.nodes[node]['val'] == True) and (G.nodes[node]['real'] == True) else 0 for node in G.nodes])
    tr_nb = sum([1 if (G.nodes[node]['test'] == False) and (G.nodes[node]['val'] == False) and (G.nodes[node]['real'] == True) else 0 for node in G.nodes])
    te_nb = sum([1 if (G.nodes[node]['test'] == True) and (G.nodes[node]['real'] == True) else 0 for node in G.nodes])

    # if val_nb == 0:
    #     node_id_tr = [node for node in G.nodes if (G.nodes[node]['test'] == False) and (G.nodes[node]['val'] == False) and (G.nodes[node]['real'] == True) ]
    #     sample_id_ls = random.sample(node_id_tr, int(tr_nb * 0.05) )
    #     for n in sample_id_ls:
    #         G.nodes[n]['val'] = True

    node_id_tr_set = set([node for node in G.nodes if (G.nodes[node]['test'] == False) and (G.nodes[node]['val'] == False) and (G.nodes[node]['real'] == True) ])
    node_id_val_set = set([node for node in G.nodes if (G.nodes[node]['test'] == False) and (G.nodes[node]['val'] == True) and (G.nodes[node]['real'] == True) ])
    node_id_te_set = set([node for node in G.nodes if (G.nodes[node]['test'] == True) and (G.nodes[node]['val'] == False) and (G.nodes[node]['real'] == True) ])
    node_real_set = {node:1 for node in G.nodes if (G.nodes[node]['real'] == True) }
    # Remove connected components (cc) that does not contain real data
    for cc in list(nx.connected_components(G)):
        has_real = False # Assuming has real data first
        fake_ls = []
        for nc in cc:
            if node_real_set.setdefault(nc, 0) == 1:
                has_real = True
            else:
                fake_ls.append(nc)
        if not has_real:
            G.remove_nodes_from(list(cc))

        # if len(fake_ls) > 0 and len(fake_ls) / len(cc) < 0.1 or len(fake_ls) / len(cc) > 0.9:
        #     G.remove_nodes_from(fake_ls)

        # 🚩 Laurence 20220728 Improved filter >>>
        elif len(fake_ls) > 0 and len(fake_ls) / len(cc) < 0.1 or len(fake_ls) / len(cc) > 0.9:
            subg_cc_real_nodes = [n for n in cc if G.nodes[n]['real']==True]
            highest_degree_node = sorted([(n,G.degree(n)) for n in subg_cc_real_nodes], key=lambda x: x[1], reverse=True )[0][0]
            path_ls = []
            for n in subg_cc_real_nodes:
                path_ls.extend(nx.shortest_path(G, highest_degree_node, n))
            mini_cc_has_all_real_nodes = G.subgraph(set(path_ls))
            nodes2remove = set(cc) - set(mini_cc_has_all_real_nodes.nodes)
            G.remove_nodes_from(nodes2remove)
        # 🚩 Laurence 20220728 Improved filter <<<


        # 🚩 Laurence 20220728 Double safty to remove 0-degree nodes >>>
        if remove_isolated_nodes:
            n_isolated_ls = [n for n in G.nodes if G.degree(n)==0]
            G.remove_nodes_from(n_isolated_ls)
        # 🚩 Laurence 20220728 Double safty to remove 0-degree nodes <<<


    ## Make sure the graph has edge train_removed annotations
    ## (some datasets might already have this..)
    print("Loaded data.. now preprocessing..")
    for edge in G.edges():
        if (G.nodes[edge[0]]['val'] or G.nodes[edge[1]]['val'] or
            G.nodes[edge[0]]['test'] or G.nodes[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False

    if normalize and not feats is None:
        from sklearn.preprocessing import StandardScaler
        train_ids = np.array([id_map[n] for n in list(G.nodes()) if not G.nodes[n]['val'] and not G.nodes[n]['test']])
        train_feats = feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)
    
    if load_walks:
        with open(prefix + "-walks.txt") as fp:
            for line in fp:
                walks.append(map(conversion, line.split()))

    return G, feats, id_map, walks, class_map

def run_random_walks(G, nodes, num_walks=N_WALKS):
    pairs = []
    for count, node in enumerate(nodes):
        if G.degree(node) == 0:
            continue
        for i in range(num_walks):
            curr_node = node
            for j in range(WALK_LEN):
                next_node = random.choice(G.neighbors(curr_node))
                # self co-occurrences are useless
                if curr_node != node:
                    pairs.append((node,curr_node))
                curr_node = next_node
        if count % 1000 == 0:
            print("Done walks for", count, "nodes")
    return pairs

if __name__ == "__main__":
    """ Run random walks """
    graph_file = sys.argv[1]
    out_file = sys.argv[2]
    G_data = json.load(open(graph_file))
    G = json_graph.node_link_graph(G_data)
    nodes = [n for n in list(G.nodes()) if not G.nodes[n]["val"] and not G.nodes[n]["test"]]
    G = G.subgraph(nodes)
    pairs = run_random_walks(G, nodes)
    with open(out_file, "w") as fp:
        fp.write("\n".join([str(p[0]) + "\t" + str(p[1]) for p in pairs]))
