import heapq
from os import times_result
import pickle
from math import sqrt
from collections import deque
import copy
# from FirmCore.FirmCore import FirmCore_k
# from Utils.density import density, density_log



def FirmTruss_k(multilayer_graph, nodes_iterator, layers_iterator, k, lammy, query_nodes, information, save=False):

    # support of each edge schema in each layer
    delta = {}

    # edge schemas with the same top lambda support
    delta_map = {}

    # support layers
    sup_layers = {}
    
    nodes = FirmCore_k(multilayer_graph, nodes_iterator, layers_iterator, k-1, lammy, query_nodes, information)
    flag = [False for node in nodes_iterator]

    for node in nodes:
        flag[node] = True

    for node in nodes:
        for layer, layer_neighbors in enumerate(multilayer_graph[node]):
            for neighbor in layer_neighbors:
                if flag[neighbor] and neighbor > node:
                     delta[(node, neighbor)] = [0 for layer in layers_iterator]
                     sup_layers[(node, neighbor)] = set()

    for node in nodes:
        for layer, layer_neighbors in enumerate(multilayer_graph[node]):
            for neighbor in layer_neighbors:
                if flag[neighbor] and neighbor > node:
                    common_neighbors = set(multilayer_graph[node][layer]) & set(multilayer_graph[neighbor][layer])
                    for neighbor2 in common_neighbors:
                        if flag[neighbor2] and neighbor2 > neighbor:                            
                            delta[(node, neighbor)][layer] += 1
                            delta[(node, neighbor2)][layer] += 1
                            delta[(neighbor, neighbor2)][layer] += 1
                            sup_layers[(node, neighbor)].add(layer)
                            sup_layers[(node, neighbor2)].add(layer)
                            sup_layers[(neighbor, neighbor2)].add(layer)



    # edge schema set
    edge_schema_set = list(delta.keys())

    B = [set(), set()]

    for edgeS in edge_schema_set:
        delta_map[edgeS] = heapq.nlargest(lammy, delta[edgeS])[-1] + 2
        B[delta_map[edgeS] >= k].add(edgeS)


    while B[0]:
        edgeS = B[0].pop()
        (u, v) = edgeS
        delta_map[edgeS] = 0
        neighbors = set()
        for layer, layer_neighbors in enumerate(multilayer_graph[u]):
            if layer in sup_layers[(u, v)]:
                for neighbor in layer_neighbors:
                    if flag[neighbor]:
                        if neighbor in multilayer_graph[v][layer]:
                            if delta_map[(min(u, neighbor), max(u, neighbor))] > k:
                                delta[(min(neighbor, u), max(neighbor, u))][layer] -= 1
                                if delta[(min(neighbor, u), max(neighbor, u))][layer] + 3 == delta_map[(min(neighbor, u), max(neighbor, u))]:
                                    neighbors.add((min(neighbor, u), max(neighbor, u)))

                            if delta_map[(min(v, neighbor), max(v, neighbor))] > k:
                                delta[(min(neighbor, v), max(neighbor, v))][layer] -= 1
                                if delta[(min(neighbor, v), max(neighbor, v))][layer] + 3 == delta_map[(min(neighbor, v), max(neighbor, v))]:
                                    neighbors.add((min(neighbor, v), max(neighbor, v)))

        for edgeS2 in neighbors:
            delta_map[edgeS2] = heapq.nlargest(lammy, delta[edgeS2])[-1] + 2
            if delta_map[edgeS2] < k:
                B[1].remove(edgeS2)
                B[0].add(edgeS2)
        
    # end of the algorithm
    information.end_algorithm()

    q = query_nodes[0]
    nodes = [False for node in nodes_iterator]
    nodes[q] = True
    bfs = deque([q])

    while bfs:
        u = bfs.popleft()
        for layer, layer_neighbors in enumerate(multilayer_graph[u]):
            for neighbor in layer_neighbors:
                if neighbor in B[1] and not nodes[neighbor]:
                    nodes[neighbor] = True
                    bfs.append(neighbor)
    community = set()

    for node in nodes_iterator:
        if nodes[node]:
            community.add(node)

    if save:
        a_file = open("./output/" + information.dataset + "_" + str(lammy) + "_FirmTruss_decomposition.pkl", "wb")
        pickle.dump(community, a_file)
        a_file.close()

    return community



############################################################
############################################################
############################################################
##################### Decomposition ########################
############################################################
############################################################
############################################################


# Given lambda

def FirmTruss(multilayer_graph, nodes_iterator, layers_iterator, lammy, information, save=False):
     # support of each edge schema in each layer
    delta = {}

    # edge schemas with the same top lambda support
    delta_map = {}

    k_start = 2

    if lammy == 1:
        k_start = 3

    for node in nodes_iterator:
        for layer, layer_neighbors in enumerate(multilayer_graph[node]):
            for neighbor in layer_neighbors:
                if neighbor > node:
                    if (node, neighbor) not in delta:
                        delta[(node, neighbor)] = [0 for layer in layers_iterator]
                    
                    common_neighbors = set(multilayer_graph[node][layer]) & set(multilayer_graph[neighbor][layer])

                    for neighbor2 in common_neighbors:
                        if neighbor2 > neighbor:
                            if (neighbor, neighbor2) not in delta:
                                delta[(neighbor, neighbor2)] = [0 for layer in layers_iterator]
                            
                            delta[(node, neighbor)][layer] += 1
                            delta[(node, neighbor2)][layer] += 1
                            delta[(neighbor, neighbor2)][layer] += 1


    # edge schema set
    edge_schema_set = list(delta.keys())

    information.start_algorithm()

    for edgeS in edge_schema_set:
        delta_map[edgeS] = heapq.nlargest(lammy, delta[edgeS])[-1]

    
    k_max = max(list(delta_map.values())) + 2


    B = [set() for _ in range(k_max + 1)]

    for edgeS in edge_schema_set:
        B[delta_map[edgeS] + 2].add(edgeS)


    print("maximum k = ", k_max)

    for k in range(k_start, k_max + 1):
        while B[k]:
            edgeS = B[k].pop()
            delta_map[edgeS] = k
            (u, v) = edgeS
            neighbors = set()

            for layer, layer_neighbors in enumerate(multilayer_graph[u]):
                for neighbor in layer_neighbors:
                    if delta_map[(min(u, neighbor), max(u, neighbor))] > k - 2:
                        delta[(min(neighbor, u), max(neighbor, u))][layer] -= 1
                        delta[(min(neighbor, v), max(neighbor, v))][layer] -= 1
                        if delta[(min(neighbor, u), max(neighbor, u))][layer] + 1 == delta_map[(min(neighbor, u), max(neighbor, u))]:
                            neighbors.add((min(neighbor, u), max(neighbor, u)))
                        if delta[(min(neighbor, v), max(neighbor, v))][layer] + 1 == delta_map[(min(neighbor, v), max(neighbor, v))]:
                            neighbors.add((min(neighbor, v), max(neighbor, v)))

            for edgeS2 in neighbors:
                B[delta_map[edgeS2]].remove(edgeS2)
                delta_map[edgeS2] = heapq.nlargest(lammy, delta[edgeS2])[-1]
                B[max(delta_map[edgeS2], k)].add(edgeS2)
        
    # end of the algorithm
    information.end_algorithm()

    if save:
        a_file = open("./output/" + information.dataset + "_" + str(lammy) + "_FirmTruss_decomposition.pkl", "wb")
        pickle.dump(delta_map, a_file)
        a_file.close()

    return 






def FirmTruss_decomposition(multilayer_graph, nodes_iterator, layers_iterator, information, save=False):
    dist_trusses = 0
    # support of each edge schema in each layer
    delta_main = {}

    # support layers
    sup_layers = {}

    for node in nodes_iterator:
            for layer, layer_neighbors in enumerate(multilayer_graph[node]):
                for neighbor in layer_neighbors:
                    if neighbor > node:
                        delta_main[(node, neighbor)] = [0 for layer in layers_iterator]
                        sup_layers[(node, neighbor)] = set()

    for node in nodes_iterator:
            for layer, layer_neighbors in enumerate(multilayer_graph[node]):
                for neighbor in layer_neighbors:
                    if neighbor > node:  
                        common_neighbors = set(multilayer_graph[node][layer]) & set(multilayer_graph[neighbor][layer])
                        for neighbor2 in common_neighbors:
                            if neighbor2 > neighbor:
                                delta_main[(node, neighbor)][layer] += 1
                                delta_main[(node, neighbor2)][layer] += 1
                                delta_main[(neighbor, neighbor2)][layer] += 1
                                sup_layers[(node, neighbor)].add(layer)
                                sup_layers[(node, neighbor2)].add(layer)
                                sup_layers[(neighbor, neighbor2)].add(layer)

    # edge schema set
    edge_schema_set = list(delta_main.keys())

    for lammy in range(1, max(layers_iterator) + 2):
        print("-------------- lambda = %d --------------"%lammy)
       # support of each edge schema in each layer
        delta = copy.deepcopy(delta_main)

        # edge schemas with the same top lambda support
        delta_map = {}

        k_start = 2

        if lammy == 1:
            k_start = 3


        information.start_algorithm()

        for edgeS in edge_schema_set:
            delta_map[edgeS] = heapq.nlargest(lammy, delta[edgeS])[-1] + 2

        
        k_max = max(list(delta_map.values()))


        B = [set() for _ in range(k_max + 1)]

        for edgeS in edge_schema_set:
            B[delta_map[edgeS]].add(edgeS)

        print("maximum k = ", k_max)

        for k in range(k_start, k_max + 1):
            if k_max == 2:
                dist_trusses += 1
                break
            if B[k]:
                dist_trusses += 1
            while B[k]:
                edgeS = B[k].pop()
                delta_map[edgeS] = k
                (u, v) = edgeS
                neighbors = set()

                for layer, layer_neighbors in enumerate(multilayer_graph[u]):
                    if layer in sup_layers[(u, v)]:
                        for neighbor in layer_neighbors:
                            if neighbor in multilayer_graph[v][layer]:
                                if (delta_map[(min(u, neighbor), max(u, neighbor))] > k):
                                    delta[(min(neighbor, u), max(neighbor, u))][layer] -= 1
                                    if delta[(min(neighbor, u), max(neighbor, u))][layer] + 3 == delta_map[(min(neighbor, u), max(neighbor, u))]:
                                        neighbors.add((min(neighbor, u), max(neighbor, u)))

                                if (delta_map[(min(v, neighbor), max(v, neighbor))] > k):
                                    delta[(min(neighbor, v), max(neighbor, v))][layer] -= 1
                                    if delta[(min(neighbor, v), max(neighbor, v))][layer] + 3 == delta_map[(min(neighbor, v), max(neighbor, v))]:
                                        neighbors.add((min(neighbor, v), max(neighbor, v)))

                for edgeS2 in neighbors:
                    B[delta_map[edgeS2]].remove(edgeS2)
                    delta_map[edgeS2] = heapq.nlargest(lammy, delta[edgeS2])[-1] + 2
                    B[max(delta_map[edgeS2], k)].add(edgeS2)
            
        # end of the algorithm
        information.end_algorithm()
        information.print_end_algorithm()

        if save:
            a_file = open("./output/" + information.dataset + "_" + str(lammy) + "_FirmTruss_decomposition.pkl", "wb")
            pickle.dump(delta_map, a_file)
            a_file.close()
        
        print("Number of Distinct Trusses = %s"%dist_trusses)


    print("Number of Distinct Trusses = %s"%dist_trusses)
    return











###############################################################
###############################################################
###############################################################
##################### Community Search ########################
###############################################################
###############################################################
###############################################################









