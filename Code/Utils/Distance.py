from array import array


def get_distances(multilayer_graph, nodes, nodes_iterator, layers_iterator, query_nodes):
    q = int(query_nodes)

    parent = dict()
    que = [0 for _ in nodes_iterator]

    distance = {}

    for node in nodes:
        distance[node] = 10000000
     
    front, rear = -1, 0
    visited = [False for _ in nodes_iterator]
    visited[q] = True
    parent[q] = (q, 1)
    que[rear] = q
    distance[q] = 0

    front += 1

    for layer, layer_neighbors in enumerate(multilayer_graph[q]):
        for neighbor in (nodes & set(layer_neighbors)):
            if not visited[neighbor]:
                rear += 1
                que[rear] = neighbor
                visited[neighbor] = True
                parent[neighbor] = (q, layer)
                distance[neighbor] = 1
 
    while front != rear:
        front += 1
        u = que[front]

        for layer, layer_neighbors in enumerate(multilayer_graph[u]):
            for neighbor in (nodes & set(layer_neighbors)):
                if not visited[neighbor]:
                    rear += 1
                    que[rear] = neighbor
                    visited[neighbor] = True
                    parent[neighbor] = (u, layer)
                    if layer == parent[u][1]:
                        distance[neighbor] = distance[u] + 1
                    else:
                        distance[neighbor] = distance[u] + 2
    
    return distance


def get_d_hub_neighborhood(multilayer_graph, nodes_iterator, d, layers_iterator, query_nodes, SFT_index=False, k=0):
    q = int(query_nodes)

    parent = [0 for _ in nodes_iterator]
    que = [0 for _ in nodes_iterator]

    distance = {}
     
    front, rear = -1, 0
    visited = [False for _ in nodes_iterator]
    visited[q] = True
    parent[q] = (q, 1)
    que[rear] = q
    distance[q] = 0

    front += 1
    if not SFT_index:
        for layer, layer_neighbors in enumerate(multilayer_graph[q]):
            for neighbor in layer_neighbors:
                if not visited[neighbor]:
                    rear += 1
                    que[rear] = neighbor
                    visited[neighbor] = True
                    parent[neighbor] = (q, layer)
                    distance[neighbor] = 1

        if d > 1:
            while front != rear:
                front += 1
                u = que[front]
                if distance[u] < d:
                    for layer, layer_neighbors in enumerate(multilayer_graph[u]):
                        for neighbor in layer_neighbors:
                            if not visited[neighbor]:
                                rear += 1
                                que[rear] = neighbor
                                visited[neighbor] = True
                                parent[neighbor] = (u, layer)
                                if layer == parent[u][1]:
                                    distance[neighbor] = distance[u] + 1
                                else:
                                    distance[neighbor] = distance[u] + 2
        
        new_multilayer_graph = dict({})

        selected_nodes = set(distance.keys())

        for node in selected_nodes:
            new_multilayer_graph[node] = [array('i') for _ in layers_iterator]
            for layer, layer_neighbors in enumerate(multilayer_graph[node]):
                common_selected_node = (selected_nodes & set(layer_neighbors))
                for neighbour in common_selected_node:
                    new_multilayer_graph[node][layer].append(neighbour)

    else:
        for layer, layer_neighbors in enumerate(multilayer_graph[q]):
            for neighbor in layer_neighbors:
                if not visited[neighbor] and SFT_index[(min(q, neighbor), max(q, neighbor))] >= k:
                    rear += 1
                    que[rear] = neighbor
                    visited[neighbor] = True
                    parent[neighbor] = (q, layer)
                    distance[neighbor] = 1

        if d > 1:
            while front != rear:
                front += 1
                u = que[front]
                if distance[u] < d:
                    for layer, layer_neighbors in enumerate(multilayer_graph[u]):
                        for neighbor in layer_neighbors:
                            if SFT_index[(min(u, neighbor), max(u, neighbor))] >= k:
                                if not visited[neighbor]:
                                    rear += 1
                                    que[rear] = neighbor
                                    visited[neighbor] = True
                                    parent[neighbor] = (u, layer)
                                    if layer == parent[u][1]:
                                        distance[neighbor] = distance[u] + 1
                                    else:
                                        distance[neighbor] = distance[u] + 2
        
        new_multilayer_graph = dict({})

        selected_nodes = set(distance.keys())

        for node in selected_nodes:
            new_multilayer_graph[node] = [array('i') for _ in layers_iterator]
            for layer, layer_neighbors in enumerate(multilayer_graph[node]):
                common_selected_node = (selected_nodes & set(layer_neighbors))
                for neighbour in common_selected_node:
                    new_multilayer_graph[node][layer].append(neighbour)

    return new_multilayer_graph, distance

def update_distance(multilayer_graph, nodes, nodes_iterator, layers_iterator, query_nodes, distances):
    q = query_nodes[0]

