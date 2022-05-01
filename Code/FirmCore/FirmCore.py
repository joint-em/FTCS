import heapq
from os import times_result
import pickle
from math import sqrt
from collections import deque
import copy
from Utils.Distance import update_distance, get_distances
from array import array
import gc




def FirmCore_k(multilayer_graph, nodes_iterator, layers_iterator, k, lammy, query_nodes, information, save=False):
    # degree of each node in each layer
    delta = {}

    # nodes with the same top lambda degree
    delta_map = {}

    # set of neighbors that we need to update
    neighbors = set()

    q = int(query_nodes)

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

    for node in nodes_iterator:
        if nodes[node]:
            delta_map[node] = heapq.nlargest(lammy, delta[node])[-1]
            B[delta_map[node] >= k].add(node)

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

    
    new_multilayer_graph = dict({})
    number_of_edges = 0

    for node in community:
        new_multilayer_graph[node] = [array('i') for _ in layers_iterator]
        for layer, layer_neighbors in enumerate(multilayer_graph[node]):
            for neighbour in layer_neighbors:
                if nodes[neighbour]:
                    new_multilayer_graph[node][layer].append(neighbour)
                    number_of_edges += 1

    del(multilayer_graph)
    gc.collect()
    print("Number of nodes with degree at least " + str(k) + ":", len(community))
    # print(community)
    # for node in community:
    #     print(delta_map[node],  end=", ")
    # print()
    return community, delta, new_multilayer_graph



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
        a_file = open("./output/" + information.dataset + "_" + str(lammy) + "_FirmCore_decomposition.pkl", "wb")
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

        max_core = 0

        if lammy == 1:
            k_start = 1

        information.start_algorithm()

        for node in nodes_iterator:
            delta[node] = [len([neighbor for neighbor in multilayer_graph[node][layer]]) for layer in layers_iterator]
            delta_map[node] = heapq.nlargest(lammy, delta[node])[-1]

        if save:
            a_file = open("./output/" + information.dataset + "_" + str(lammy) + "_FirmCore_decomposition_degree.pkl", "wb")
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
                max_core = k
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

        print("Maximum coreness = %s"%max_core)

        if save:
            a_file = open("./output/" + information.dataset + "_" + str(lammy) + "_FirmCore_decomposition.pkl", "wb")
            pickle.dump(delta_map, a_file)
            a_file.close()






###############################################################
###############################################################
###############################################################
##################### Community Search ########################
###############################################################
###############################################################
###############################################################



def FirmCore_Global(multilayer_graph, nodes_iterator, layers_iterator, k, lammy, query_nodes, information, save=False):

    information.start_algorithm()

    G_0, delta, new_multilayer_graph = FirmCore_k(multilayer_graph, nodes_iterator, layers_iterator, k, lammy, query_nodes, information)

    print("G0 size:", len(G_0))

    flag = [False for node in nodes_iterator]

    for node in G_0:
        flag[node] = True

    graph_query_diam = 10000000
    community = set()

    while query_nodes in G_0:
        distances = get_distances(new_multilayer_graph, G_0, nodes_iterator, layers_iterator, query_nodes)
        max_distance = max(distances.values())
        if max_distance < graph_query_diam and G_0 != set([query_nodes]):
            graph_query_diam = max_distance
            community = copy.deepcopy(G_0)

        G = copy.deepcopy(G_0)

        for node in G:
            if distances[node] == max_distance and flag[node]:
                G_0.remove(node)
                flag[node] = False
                neighbors = set()

                for layer, layer_neighbors in enumerate(new_multilayer_graph[node]):
                    for neighbor in layer_neighbors:
                        if flag[neighbor]:
                            delta[neighbor][layer] -= 1
                            if heapq.nlargest(lammy, delta[neighbor])[-1] < k:
                                neighbors.add(neighbor)
                               

                while neighbors:
                    u = neighbors.pop()
                    G_0.remove(u)
                    flag[u] = False
                                        
                    for layer, layer_neighbors in enumerate(new_multilayer_graph[u]):
                        for neighbor in layer_neighbors:
                            if flag[neighbor]:
                                delta[neighbor][layer] -= 1
                                if heapq.nlargest(lammy, delta[neighbor])[-1] < k:
                                    neighbors.add(neighbor)
                                

    information.end_algorithm()
    information.print_end_algorithm()

    q = int(query_nodes)
    nodes = [False for node in nodes_iterator]
    nodes[q] = True
    bfs = deque([q])

    while bfs:
        u = bfs.popleft()
        for layer, layer_neighbors in enumerate(new_multilayer_graph[u]):
            for neighbor in layer_neighbors:
                if neighbor in community and not nodes[neighbor]:
                    nodes[neighbor] = True
                    bfs.append(neighbor)
    connected_community = set()

    for node in community:
        if nodes[node]:
            connected_community.add(node)

    print("Community Size:", len(connected_community))
    # print("Community: ", connected_community)
    print("Query Distance: ", graph_query_diam)
    return connected_community


