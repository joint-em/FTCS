import heapq
from os import times_result
import pickle
from math import sqrt
from collections import deque
# from ..Utils.density import density, density_log



def FirmCore_k(multilayer_graph, nodes_iterator, layers_iterator, k, lammy, query_nodes, information, save=False):
    # degree of each node in each layer
    delta = {}

    # nodes with the same top lambda degree
    delta_map = {}

    # set of neighbors that we need to update
    neighbors = set()

    q = query_nodes[0]

    nodes = [False for node in nodes_iterator]
    nodes[q] = True
    bfs = deque([q])

    while bfs:
        u = bfs.popleft()
        delta[u] = [0 for layer in layers_iterator]
        for layer, layer_neighbors in enumerate(multilayer_graph[u]):
            for neighbor in layer_neighbors:
                delta[u][layer] += 1
                if not nodes[neighbor]:
                    nodes[neighbor] = True
                    bfs.append(neighbor)

    B = [set(), set()]

    for node in nodes:
        delta_map[node] = heapq.nlargest(lammy, delta[node])[-1]
        B[delta_map[node] >= k].add(node)

    information.start_algorithm()

    while B[0]:
        node = B[0].pop()
        delta_map[node] = 0
        neighbors = set()

        for layer, layer_neighbors in enumerate(multilayer_graph[node]):
            for neighbor in layer_neighbors:
                if delta_map[neighbor] >= k:
                    delta[neighbor][layer] -= 1
                    if delta[neighbor][layer] + 1 == delta_map[neighbor]:
                        neighbors.add(neighbor)

        for neighbor in neighbors:
            delta_map[neighbor] = heapq.nlargest(lammy, delta[neighbor])[-1]
            if delta_map[neighbor] < k:
                B[1].remove(neighbor)
                B[0].add(neighbor)
        
    # end of the algorithm
    information.end_algorithm()

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
        a_file = open("../output/" + information.dataset + "_" + str(lammy) + "_FirmCore_decomposition.pkl", "wb")
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

def FirmCore(multilayer_graph, nodes_iterator, layers_iterator, lammy, information, save=False):
    # degree of each node in each layer
    delta = {}

    # nodes with the same top lambda degree
    delta_map = {}

    # set of neighbors that we need to update
    neighbors = set()

    k_max = 0
    k_start = 0

    if lammy == 1:
        k_start = 1

    information.start_algorithm()

    for node in nodes_iterator:
        delta[node] = [len([neighbor for neighbor in multilayer_graph[node][layer]]) for layer in layers_iterator]
        delta_map[node] = heapq.nlargest(lammy, delta[node])[-1]

    k_max = max(list(delta_map.values()))
    # bin-sort for removing a vertex
    B = [set() for i in range(k_max + 1)]

    for node in nodes_iterator:
        B[delta_map[node]].add(node)

    print("maximum k = ", k_max)
    for k in range(k_start, k_max + 1):
        while B[k]:
            node = B[k].pop()
            delta_map[node] = k
            neighbors = set()

            for layer, layer_neighbors in enumerate(multilayer_graph[node]):
                for neighbor in layer_neighbors:
                    if delta_map[neighbor] > k:
                        delta[neighbor][layer] -= 1
                        if delta[neighbor][layer] + 1 == delta_map[neighbor]:
                            neighbors.add(neighbor)

            for neighbor in neighbors:
                B[delta_map[neighbor]].remove(neighbor)
                delta_map[neighbor] = heapq.nlargest(lammy, delta[neighbor])[-1]
                B[max(delta_map[neighbor], k)].add(neighbor)
        
    # end of the algorithm
    information.end_algorithm(max(delta_map.values()))

    if save:
        a_file = open("../output/" + information.dataset + "_" + str(lammy) + "_FirmCore_decomposition.pkl", "wb")
        pickle.dump(delta_map, a_file)
        a_file.close()

    return





def FirmCore_decomposition(multilayer_graph, nodes_iterator, layers_iterator, information, save=False):
    for lammy in range(1, max(layers_iterator) + 2):
        print("-------------- lambda = %d --------------"%lammy)
        # degree of each node in each layer
        delta = {}

        # nodes with the same top lambda degree
        delta_map = {}

        # set of neighbors that we need to update
        neighbors = set()

        k_max = 0
        k_start = 0

        # distinct cores
        dist_cores = 0

        if lammy == 1:
            k_start = 1

        information.start_algorithm()

        for node in nodes_iterator:
            delta[node] = [len([neighbor for neighbor in multilayer_graph[node][layer]]) for layer in layers_iterator]
            delta_map[node] = heapq.nlargest(lammy, delta[node])[-1]

        if save:
            a_file = open("../output/" + information.dataset + "_" + str(lammy) + "_FirmCore_decomposition_degree.pkl", "wb")
            pickle.dump(delta_map, a_file)
            a_file.close()

        k_max = max(list(delta_map.values()))
        # bin-sort for removing a vertex
        B = [set() for i in range(k_max + 1)]

        for node in nodes_iterator:
            B[delta_map[node]].add(node)

        print("maximum k = ", k_max)
        for k in range(k_start, k_max + 1):
            if B[k]:
                dist_cores += 1
            while B[k]:
                node = B[k].pop()
                delta_map[node] = k
                neighbors = set()

                for layer, layer_neighbors in enumerate(multilayer_graph[node]):
                    for neighbor in layer_neighbors:
                        if delta_map[neighbor] > k:
                            delta[neighbor][layer] -= 1
                            if delta[neighbor][layer] + 1 == delta_map[neighbor]:
                                neighbors.add(neighbor)

                for neighbor in neighbors:
                    B[delta_map[neighbor]].remove(neighbor)
                    delta_map[neighbor] = heapq.nlargest(lammy, delta[neighbor])[-1]
                    B[max(delta_map[neighbor], k)].add(neighbor)
            
        # end of the algorithm
        information.end_algorithm()
        information.print_end_algorithm()

        print("Number of Distinct cores = %s"%dist_cores)

        if save:
            a_file = open("../output/" + information.dataset + "_" + str(lammy) + "_FirmCore_decomposition.pkl", "wb")
            pickle.dump(delta_map, a_file)
            a_file.close()






