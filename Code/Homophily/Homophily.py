from cmath import inf
from FirmTruss.FirmTruss import FirmTruss_k_index, FirmTruss_k
import pickle
import math
import heapq
from collections import deque
import copy


def AFTCS_Approx(multilayer_graph, attributes, nodes_iterator, layers_iterator, k, lammy, p, query_nodes, information, index=True, save=False, verbos=True):

    information.start_algorithm()

    if index:
        G_0, delta, delta_map, sup_layer, edge_schema_set = FirmTruss_k_index(multilayer_graph, nodes_iterator, layers_iterator, k, lammy, query_nodes, information)
        new_multilayer_graph = multilayer_graph
    
    else:
        G_0, delta, delta_map, sup_layer, edge_schema_set, new_multilayer_graph = FirmTruss_k(multilayer_graph, nodes_iterator, layers_iterator, k, lammy, query_nodes, information)
    

    primary_size = len(G_0)
    community = G_0
    
    h = {}

    flag = dict({})
    for edgeS in edge_schema_set:
        flag[edgeS] = True

    for node in G_0:
        h[node] = 0
        for node2 in G_0:
            h[node] += sum(i[0] * i[1] for i in zip(attributes[node], attributes[node2]))

    community_homophily_score = 0

    while query_nodes in G_0:
        homophily_score = p_mean(h, p)
        if homophily_score > community_homophily_score:
            community = copy.deepcopy(G_0)
            community_homophily_score = homophily_score

        node = marginal_gain(h, attributes, p)
        G_0.remove(node)
        h = update_scores(h, attributes, node)

        neighbors = set()
        for layer, layer_neighbors in enumerate(new_multilayer_graph[node]):
            for neighbor in set(layer_neighbors) & set(G_0):
                if flag[(min(node, neighbor), max(node, neighbor))] and layer in sup_layer[(min(node, neighbor), max(node, neighbor))]:
                    flag[(min(node, neighbor), max(node, neighbor))] = False
                    for neighbor2 in set(new_multilayer_graph[neighbor][layer]) & set(G_0) & set(new_multilayer_graph[node][layer]):
                        if neighbor2 > neighbor:
                            flag[(min(node, neighbor2), max(node, neighbor2))] = False
                            delta[(neighbor, neighbor2)][layer] -= 1

                            if delta_map[(neighbor, neighbor2)] == delta[(neighbor, neighbor2)][layer] + 3:
                                delta_map[(neighbor, neighbor2)] = heapq.nlargest(lammy, delta[(neighbor, neighbor2)])[-1] + 2

                            if delta_map[(neighbor, neighbor2)] < k:
                                neighbors.add((neighbor, neighbor2))     

        while neighbors:
            edgeS = neighbors.pop()
            (u, v) = edgeS
            flag[edgeS] = False
            
            for layer, layer_neighbors in enumerate(new_multilayer_graph[u]):
                if layer in sup_layer[(edgeS)]:
                    for neighbor in set(layer_neighbors) & set(G_0) & set(new_multilayer_graph[v][layer]):
                        if flag[(min(neighbor, u), max(neighbor, u))] and flag[(min(neighbor, v), max(neighbor, v))]:
                            delta[(min(neighbor, u), max(neighbor, u))][layer] -= 1
                            delta[(min(neighbor, v), max(neighbor, v))][layer] -= 1

                            if delta_map[(min(neighbor, u), max(neighbor, u))] == delta[(min(neighbor, u), max(neighbor, u))][layer] + 3:
                                delta_map[(min(neighbor, u), max(neighbor, u))] = heapq.nlargest(lammy, delta[(min(neighbor, u), max(neighbor, u))])[-1] + 2
                                if delta_map[(min(neighbor, u), max(neighbor, u))] < k:
                                    neighbors.add((min(neighbor, u), max(neighbor, u)))
                            
                            if delta_map[(min(neighbor, v), max(neighbor, v))] == delta[(min(neighbor, v), max(neighbor, v))][layer] + 3:
                                delta_map[(min(neighbor, v), max(neighbor, v))] = heapq.nlargest(lammy, delta[(min(neighbor, v), max(neighbor, v))])[-1] + 2
                                if delta_map[(min(neighbor, v), max(neighbor, v))] < k:
                                    neighbors.add((min(neighbor, v), max(neighbor, v)))

        G = set()
        for edgeS in edge_schema_set:
            if delta_map[edgeS] >= k:
                G.add(edgeS[0])
                G.add(edgeS[1])
        
        remove_set = set(G_0) - set(G)

        for node in remove_set:
            G_0.remove(node)
            h = update_scores(h, attributes, node)


    information.end_algorithm()
    if verbos:
        information.print_end_algorithm()
        print("Community Size:", len(community))
        print("Community: ", community)
        print("Homophily Score: ", community_homophily_score)
        print("# Free Riders: ", primary_size - len(community))
    return community





def p_mean(h, p):
    score = 0
    if p <= -100:
        score = min(h.values())

    elif p >= 100:
        score = max(h.values())

    elif p == 0:
        score = 1
        for val in h.values():
            score *= math.log(val)
        score = score * (1/len(h))

    else:
        for val in h.values():
            score += val ** p
        score /= len(h)
        score = score ** (1/p)

    return score


def marginal_gain(h, attributes, p):
    if p <= -100:
        return min(h, key=h.get)
    
    else:
        subgraph = h.keys()
        delta = {}
        for node in subgraph:
            delta[node] = h[node]**p
            for node2 in subgraph:
                if node2 != node:
                    delta[node] += (h[node2]**p - (h[node2] - sum(i[0] * i[1] for i in zip(attributes[node], attributes[node2])))**p)

        if p >= 0:
            return min(delta, key=delta.get)
    
        else:                    
            return max(delta, key=delta.get)


def update_scores(h, attributes, node):
    h.pop(node)
    subgraph = h.keys()
    for node2 in subgraph:
        h[node2] -= sum(i[0] * i[1] for i in zip(attributes[node], attributes[node2]))

    return h