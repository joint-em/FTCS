import heapq
from os import times_result
import pickle
from math import sqrt
from collections import deque
import copy
from FirmCore.FirmCore import FirmCore_k
from Utils.density import density, density_log
from Utils.Distance import update_distance, get_distances, get_d_hub_neighborhood
import time


def FirmTruss_k(multilayer_graph, nodes_iterator, layers_iterator, k, lammy, query_nodes, information, save=False):

    nodes, _, new_multilayer_graph = FirmCore_k(multilayer_graph, nodes_iterator, layers_iterator, k-1, lammy, query_nodes, information)

    # support of each edge schema in each layer
    delta, sup_layer = compute_support(multilayer_graph, layers_iterator, nodes)

    # edge schemas with the same top lambda support
    delta_map = {}                        

    # edge schema set
    edge_schema_set = list(delta.keys())    

    B = [set(), set()]
    flag = dict()


    for edgeS in edge_schema_set:
        delta_map[edgeS] = heapq.nlargest(lammy, delta[edgeS])[-1] + 2
        B[delta_map[edgeS] >= k].add(edgeS)
        flag[edgeS] = True

    while B[0]:
        edgeS = B[0].pop()
        flag[edgeS] = False
        (u, v) = edgeS
        delta_map[edgeS] = 0
        neighbors = set()
        for layer, layer_neighbors in enumerate(new_multilayer_graph[u]):
            if layer in sup_layer[(u, v)]:
                for neighbor in set(layer_neighbors) & set(new_multilayer_graph[v][layer]) & nodes:
                    edgeS1 = (min(neighbor, u), max(neighbor, u))
                    edgeS2 = (min(v, neighbor), max(v, neighbor))

                    if flag[edgeS1] and flag[edgeS2]:
                        delta[edgeS1][layer] -= 1
                        if delta[edgeS1][layer] + 3 == delta_map[edgeS1]:
                            neighbors.add(edgeS1)

                        delta[edgeS2][layer] -= 1
                        if delta[edgeS2][layer] + 3 == delta_map[edgeS2]:
                            neighbors.add(edgeS2)
        
        for edgeS2 in neighbors:
            if delta_map[edgeS2] >= k:
                delta_map[edgeS2] = heapq.nlargest(lammy, delta[edgeS2])[-1] + 2
                if delta_map[edgeS2] < k:
                    B[1].remove(edgeS2)
                    B[0].add(edgeS2)
    
    selected_nodes = set()
    for edgeS in B[1]:
        selected_nodes.add(edgeS[0])
        selected_nodes.add(edgeS[1])

    community = connected_component_k(new_multilayer_graph, selected_nodes, k, delta_map, query_nodes)
    # community = connected_component(new_multilayer_graph, selected_nodes, query_nodes)


    # community = selected_nodes

    delta, sup_layer = compute_support(new_multilayer_graph, layers_iterator, community)


    if save:
        a_file = open("./output/" + information.dataset + "_" + str(lammy) + "_FirmTruss_decomposition.pkl", "wb")
        pickle.dump(community, a_file)
        a_file.close()

    return community, delta, delta_map, sup_layer, edge_schema_set, new_multilayer_graph



def FirmTruss_k_index(multilayer_graph, nodes_iterator, layers_iterator, k, lammy, query_nodes, information, save=False):
    
    information.pause_algorithm()
    index_file = open("./output/" + information.dataset + "_" + str(lammy) + "_FirmTruss_decomposition.pkl", "rb")
    delta_map = pickle.load(index_file)
    information.continue_algorithm()
    community = set()

    edge_schema_set = list(delta_map.keys())

    flag = {}
    for edgeS in edge_schema_set:
        flag[edgeS] = False
        # if delta_map[edgeS] >= k:
        #     print(edgeS)


    candidate = set()
    candidate.add(query_nodes)

    


    while candidate:
        node = candidate.pop()

        for layer, layer_neighbors in enumerate(multilayer_graph[node]):
            for neighbor in layer_neighbors:
                edgeS = (min(node, neighbor), max(node, neighbor))
                if not flag[edgeS]:
                    flag[edgeS] = True
                    if delta_map[edgeS] >= k:
                        candidate.add(neighbor)
                        community.add(node)
                        community.add(neighbor)

    if save:
        a_file = open("./output/" + information.dataset + "_" + str(lammy) + "_FirmTruss_decomposition.pkl", "wb")
        pickle.dump(community, a_file)
        a_file.close()


    delta, sup_layer = compute_support(multilayer_graph, layers_iterator, community)

    return community, delta, delta_map, sup_layer, edge_schema_set





############################################################
############################################################
############################################################
##################### Decomposition ########################
############################################################
############################################################
############################################################

def FirmTruss_decomposition(multilayer_graph, nodes_iterator, layers_iterator, information, save=False):
    dist_trusses = 0
    # support of each edge schema in each layer
    delta_main = {}

    # sup layer
    sup_layer = {}

    delta_main, sup_layer = compute_support(multilayer_graph, layers_iterator, set(nodes_iterator))


    # edge schema set
    edge_schema_set = list(delta_main.keys())

    for lammy in range(1, max(layers_iterator) + 2):
        print("-------------- lambda = %d --------------"%lammy)
       # support of each edge schema in each layer
        delta = copy.deepcopy(delta_main)

        if lammy == 1:
            previous_index = {}
            for edgeS in edge_schema_set:
                previous_index[edgeS] = 10000000

        else:
            previous_index = copy.deepcopy(delta_map)

        # edge schemas with the same top lambda support
        delta_map = {}

        k_start = 2

        if lammy == 1:
            k_start = 3


        information.start_algorithm()

        for edgeS in edge_schema_set:
            delta_map[edgeS] = min(heapq.nlargest(lammy, delta[edgeS])[-1] + 2, previous_index[edgeS])


        k_max = max(list(delta_map.values()))

        max_truss = 0


        B = [set() for _ in range(k_max + 1)]
        flag = dict()

        for edgeS in edge_schema_set:
            B[delta_map[edgeS]].add(edgeS)
            flag[edgeS] = True

        print("maximum k = ", k_max)

        for k in range(k_start, k_max + 1):
            if k_max == 2:
                max_truss = 2
                dist_trusses += 1
                break
            if B[k]:
                dist_trusses += 1
            while B[k]:
                max_truss = k
                edgeS = B[k].pop()
                flag[edgeS] = False
                delta_map[edgeS] = k
                (u, v) = edgeS
                neighbors = set()

                for layer, layer_neighbors in enumerate(multilayer_graph[u]):
                    if layer in sup_layer[(u, v)]:
                        for neighbor in set(layer_neighbors) & set(multilayer_graph[v][layer]):
                            edgeS1 = (min(u, neighbor), max(u, neighbor))
                            edgeS2 = (min(v, neighbor), max(v, neighbor))
                            if flag[edgeS1] and flag[edgeS2]:
                                delta[(min(neighbor, u), max(neighbor, u))][layer] -= 1
                                if delta[(min(neighbor, u), max(neighbor, u))][layer] + 3 == delta_map[(min(neighbor, u), max(neighbor, u))]:
                                    neighbors.add((min(neighbor, u), max(neighbor, u)))

                                delta[(min(neighbor, v), max(neighbor, v))][layer] -= 1
                                if delta[(min(neighbor, v), max(neighbor, v))][layer] + 3 == delta_map[(min(neighbor, v), max(neighbor, v))]:
                                    neighbors.add((min(neighbor, v), max(neighbor, v)))

                for edgeS2 in neighbors:
                    if flag[edgeS2] and delta_map[edgeS2] > k:
                        B[delta_map[edgeS2]].remove(edgeS2)
                        delta_map[edgeS2] = min(heapq.nlargest(lammy, delta[edgeS2])[-1] + 2, previous_index[edgeS2])
                        B[max(delta_map[edgeS2], k)].add(edgeS2)
                
        # end of the algorithm
        information.end_algorithm()
        information.print_end_algorithm()

        if save:
            a_file = open("./output/" + information.dataset + "_" + str(lammy) + "_FirmTruss_decomposition.pkl", "wb")
            pickle.dump(delta_map, a_file)
            a_file.close()

            a_file = open("./output/sup_" + information.dataset + "_" + str(lammy) + "_FirmTruss_decomposition.pkl", "wb")
            pickle.dump(delta, a_file)
            a_file.close()

            a_file = open("./output/sup_layer_" + information.dataset + "_" + str(lammy) + "_FirmTruss_decomposition.pkl", "wb")
            pickle.dump(sup_layer, a_file)
            a_file.close()
            
        
        print("Number of Distinct Trusses = %s"%max_truss)


    print("Total Number of Distinct Trusses = %s"%dist_trusses)
    return











###############################################################
###############################################################
###############################################################
##################### Community Search ########################
###############################################################
###############################################################
###############################################################


def FirmTruss_Global(multilayer_graph, nodes_iterator, layers_iterator, k, lammy, query_nodes, information, index=True, save=False, verbos=True):

    information.start_algorithm()
    if index:
        G_0, delta, delta_map, sup_layer, edge_schema_set = FirmTruss_k_index(multilayer_graph, nodes_iterator, layers_iterator, k, lammy, query_nodes, information)
        if verbos:
            print("G_0 size:", len(G_0))
        new_multilayer_graph = multilayer_graph
    
    else:
        G_0, delta, delta_map, sup_layer, edge_schema_set, new_multilayer_graph = FirmTruss_k(multilayer_graph, nodes_iterator, layers_iterator, k, lammy, query_nodes, information)
        if verbos:
            print("G_0 size:", len(G_0))

    primary_size = len(G_0)
    graph_query_diam = 10000000
    community = set()
    flag = dict({})

    for edgeS in edge_schema_set:
        flag[edgeS] = True

    while query_nodes in G_0:
        distances = get_distances(new_multilayer_graph, G_0, nodes_iterator, layers_iterator, query_nodes)
        G = copy.deepcopy(G_0)

        for node in G:
            if distances[node] >= (10000000 - 1):
                G_0.remove(node)
                distances[node] = -1
        
        max_distance = max(distances.values())
        if max_distance < graph_query_diam and G_0 != set([query_nodes]):
            graph_query_diam = max_distance
            community = copy.deepcopy(G_0)

        G = copy.deepcopy(G_0)

        for node in G:
            if distances[node] == max_distance:
                G_0.remove(node)
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
                        if layer in sup_layer[edgeS]:
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

        G_0 = G & set(G_0)


    information.end_algorithm()
    if verbos:
        information.print_end_algorithm()

    connected_community = connected_component(new_multilayer_graph, community, query_nodes)

    if verbos:
        print("Community Size:", len(connected_community))
        print("Community: ", check_FirmTruss(new_multilayer_graph, nodes_iterator, layers_iterator, k, lammy, set(connected_community)))
        print("Query Distance: ", graph_query_diam)
        print("# Free riders: ", primary_size - len(connected_community))

    return connected_community










def FirmTruss_Local(multilayer_graph, nodes_iterator, layers_iterator, k, lammy, query_nodes, information, index=False, save=False, verbos=True):

    N = len(nodes_iterator)
    stop = False

    if index:
        index_file = open("./output/" + information.dataset + "_" + str(lammy) + "_FirmTruss_decomposition.pkl", "rb")
        SFT_index = pickle.load(index_file)

    else:
        SFT_index = False
    information.start_algorithm()

    i = 1

    while i < 20:
        new_multilayer_graph, distances = get_d_hub_neighborhood(multilayer_graph, nodes_iterator, i, layers_iterator, query_nodes, SFT_index, k=k)
        V = set(distances.keys())

        if len(V) == N:
            break

        # support of each edge schema in each layer
        delta, sup_layer = compute_support(new_multilayer_graph, layers_iterator, V)

        # edge schemas with the same top lambda support
        delta_map = {}

        # edge schema set
        edge_schema_set = list(delta.keys())

        B = [set(), set()]
        flag = dict()

        for edgeS in edge_schema_set:
            delta_map[edgeS] = heapq.nlargest(lammy, delta[edgeS])[-1] + 2
            B[delta_map[edgeS] >= k].add(edgeS)
            flag[edgeS] = True

        while B[0]:
            edgeS = B[0].pop()
            flag[edgeS] = False
            (u, v) = edgeS
            delta_map[edgeS] = 0
            neighbors = set()
            for layer, layer_neighbors in enumerate(new_multilayer_graph[u]):
                if layer in sup_layer[edgeS]:
                    for neighbor in set(layer_neighbors) & set(new_multilayer_graph[v][layer]) & V:
                        edgeS1 = (min(neighbor, u), max(neighbor, u))
                        edgeS2 = (min(v, neighbor), max(v, neighbor))

                        if flag[edgeS1] and flag[edgeS2]:
                            delta[edgeS1][layer] -= 1
                            if delta[edgeS1][layer] + 3 == delta_map[edgeS1]:
                                neighbors.add(edgeS1)
                            
                            delta[edgeS2][layer] -= 1
                            if delta[edgeS2][layer] + 3 == delta_map[edgeS2]:
                                neighbors.add(edgeS2)
                                
            for edgeS2 in neighbors:
                if delta_map[edgeS2] >= k:
                    delta_map[edgeS2] = heapq.nlargest(lammy, delta[edgeS2])[-1] + 2
                    if delta_map[edgeS2] < k:
                        B[1].remove(edgeS2)
                        B[0].add(edgeS2)
        
        selected_nodes = set()

        for edgeS in B[1]:
            selected_nodes.add(edgeS[0])
            selected_nodes.add(edgeS[1])
        
        while selected_nodes and (query_nodes in selected_nodes):
            distances = get_distances(new_multilayer_graph, selected_nodes, nodes_iterator, layers_iterator, query_nodes)
            to_be_removed_vertices = [node for node in selected_nodes if distances[node] > i]
            
            if not to_be_removed_vertices:
                information.end_algorithm()
                if verbos:
                    information.print_end_algorithm()
                    print("Community Size:", len(selected_nodes))
                    print("Community: ", selected_nodes)
                    print("Query Distance: ", i)
                return selected_nodes
            
            else:
                for node in to_be_removed_vertices:
                    selected_nodes.remove(node)
                    neighbors = set()

                    for layer, layer_neighbors in enumerate(new_multilayer_graph[node]):
                        for neighbor in set(layer_neighbors) & selected_nodes:
                            flag[(min(node, neighbor), max(node, neighbor))] = False
                            for neighbor2 in (set(new_multilayer_graph[neighbor][layer]) & set(new_multilayer_graph[node][layer])) & selected_nodes:
                                flag[(min(node, neighbor2), max(node, neighbor2))] = False
                                if neighbor2 > neighbor:
                                    delta[(neighbor, neighbor2)][layer] -= 1

                                    if heapq.nlargest(lammy, delta[(neighbor, neighbor2)])[-1] < k - 2:
                                        neighbors.add((neighbor, neighbor2))     

                    while neighbors:
                        edgeS = neighbors.pop()
                        flag[edgeS] = False
                        (u, v) = edgeS
                        
                        for layer, layer_neighbors in enumerate(new_multilayer_graph[u]):
                            if layer in sup_layer[(u,  v)]:
                                for neighbor in set(layer_neighbors) & set(new_multilayer_graph[v][layer]) & selected_nodes:
                                    edgeS1 = (min(neighbor, u), max(neighbor, u))
                                    edgeS2 = (min(v, neighbor), max(v, neighbor))
                                    if flag[edgeS1] and flag[edgeS2]:
                                        delta[edgeS1][layer] -= 1
                                        delta[edgeS2][layer] -= 1

                                        delta_map_1 = heapq.nlargest(lammy, delta[edgeS1])[-1]
                                        delta_map_2 = heapq.nlargest(lammy, delta[edgeS2])[-1]

                                        if delta[edgeS1][layer] + 1 == delta_map_1 and delta_map_1 < (k - 2):
                                            neighbors.add(edgeS1)
                                        
                                        if delta[edgeS2][layer] + 1 == delta_map_2 and delta_map_2 < (k - 2):
                                            neighbors.add(edgeS2)
            
            selected_nodes = set()
            for edgeS in flag.keys():
                if flag[edgeS]:
                    selected_nodes.add(edgeS[0])
                    selected_nodes.add(edgeS[1])

        i += 1

    information.end_algorithm()
    if verbos:
        information.print_end_algorithm()

    return None









###############################################################
###############################################################
###############################################################
######################### Utils ###############################
###############################################################
###############################################################
###############################################################


def check_FirmTruss(multilayer_graph, nodes_iterator, layers_iterator, k, lammy, subgraph):
     # support of each edge schema in each layer
    delta, sup_layer = compute_support(multilayer_graph, layers_iterator, subgraph)

    # edge schemas with the same top lambda support
    delta_map = {}          
    
    # edge schema set
    edge_schema_set = list(delta.keys())

    B = [set(), set()]
    flag = dict()


    for edgeS in edge_schema_set:
        delta_map[edgeS] = heapq.nlargest(lammy, delta[edgeS])[-1] + 2
        B[delta_map[edgeS] >= k].add(edgeS)
        flag[edgeS] = True

    while B[0]:
        edgeS = B[0].pop()
        flag[edgeS] = False
        (u, v) = edgeS
        delta_map[edgeS] = 0
        neighbors = set()
        for layer, layer_neighbors in enumerate(multilayer_graph[u]):
            if layer in sup_layer[(u, v)]:
                for neighbor in (set(layer_neighbors) & subgraph & set(multilayer_graph[v][layer])):
                    edgeS1 = (min(neighbor, u), max(neighbor, u))
                    edgeS2 = (min(v, neighbor), max(v, neighbor))

                    if flag[edgeS1] and flag[edgeS2]:
                        delta[edgeS1][layer] -= 1
                        if delta[edgeS1][layer] + 3 == delta_map[edgeS1]:
                            neighbors.add(edgeS1)

                        delta[edgeS2][layer] -= 1
                        if delta[edgeS2][layer] + 3 == delta_map[edgeS2]:
                            neighbors.add(edgeS2)

        for edgeS2 in neighbors:
            if delta_map[edgeS2] >= k:
                delta_map[edgeS2] = heapq.nlargest(lammy, delta[edgeS2])[-1] + 2
                if delta_map[edgeS2] < k:
                    B[1].remove(edgeS2)
                    B[0].add(edgeS2)

    
    selected_nodes = set()

    for edgeS in B[1]:
        selected_nodes.add(edgeS[0])
        selected_nodes.add(edgeS[1])


    return selected_nodes





def compute_support(multilayer_graph, layers_iterator, subgraph):
    # support of each edge schema in each layer
    delta = {}

    # sup layer
    sup_layer = {}

    for node in subgraph:
        for layer, layer_neighbors in enumerate(multilayer_graph[node]):
            for neighbor in (set(layer_neighbors) & subgraph):
                if neighbor > node:
                    delta[(node, neighbor)] = [0 for layer in layers_iterator]
                    sup_layer[(node, neighbor)] = set()


    for node in subgraph:
        for layer, layer_neighbors in enumerate(multilayer_graph[node]):
            for neighbor in (set(layer_neighbors) & subgraph):
                if neighbor > node:
                    common_neighbors = set(multilayer_graph[node][layer]) & set(multilayer_graph[neighbor][layer]) & subgraph 
                    for neighbor2 in common_neighbors:
                        if neighbor2 > neighbor:                            
                            delta[(node, neighbor)][layer] += 1
                            delta[(node, neighbor2)][layer] += 1
                            delta[(neighbor, neighbor2)][layer] += 1
                            sup_layer[(node, neighbor)].add(layer)
                            sup_layer[(node, neighbor2)].add(layer)
                            sup_layer[(neighbor, neighbor2)].add(layer)

    
    return delta, sup_layer


def connected_component(multilayer_graph, subgraph, query_nodes):
    q = int(query_nodes)

    if q in subgraph:
        nodes = {}
        for node in subgraph:
            nodes[node] = False
        nodes[q] = True
        bfs = deque([q])

        while bfs:
            u = bfs.popleft()
            for layer, layer_neighbors in enumerate(multilayer_graph[u]):
                for neighbor in set(layer_neighbors) & subgraph:
                    if not nodes[neighbor]:
                        nodes[neighbor] = True
                        bfs.append(neighbor)
        connected_community = set()

        for node in subgraph:
            if nodes[node]:
                connected_community.add(node)
    
    else:
        connected_community = {}
    
    return connected_community



def connected_component_k(multilayer_graph, subgraph, k, delta_map, query_nodes):
    q = int(query_nodes)

    if q in subgraph:
        nodes = {}
        for node in subgraph:
            nodes[node] = False
        nodes[q] = True
        bfs = deque([q])

        while bfs:
            u = bfs.popleft()
            for layer, layer_neighbors in enumerate(multilayer_graph[u]):
                for neighbor in set(layer_neighbors) & subgraph:
                    if not nodes[neighbor] and delta_map[(min(u, neighbor), max(u, neighbor))] >= k:
                        nodes[neighbor] = True
                        bfs.append(neighbor)
        connected_community = set()

        for node in subgraph:
            if nodes[node]:
                connected_community.add(node)
    
    else:
        connected_community = {}
    
    return connected_community