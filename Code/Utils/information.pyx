from __future__ import division
from .memory_measure import memory_usage_resource
from time import time

class Information:
    def __init__(self, dataset_name=None):
        self.dataset = dataset_name    
        self.time = 0
        self.preprocess_memory = 0
        self.preprocess_time = 0
        self.execution_time_milliseconds = 0
    
    def start_algorithm(self):
        # start the algorithm
        self.execution_time_milliseconds = - self.current_milliseconds()
        self.preprocess_memory =  memory_usage_resource()

    def end_algorithm(self):
        # end the algorithm
        self.execution_time_milliseconds += self.current_milliseconds()

        self.time = self.execution_time_milliseconds / 1000.0



    def print_dataset_name(self, dataset_path):
        print('------------- Dataset -------------')
        print('Name: ' + dataset_path.split('/')[-1].capitalize())

    def print_dataset_info(self, multilayer_graph):
        number_of_nodes = multilayer_graph.number_of_nodes
        number_of_edges = multilayer_graph.get_number_of_edges()

        print('Nodes: ' + str(number_of_nodes))
        print('Edges: ' + str(number_of_edges))
        print('Layers: ' + str(multilayer_graph.number_of_layers))
        print('Edge density: ' + str((2 * number_of_edges) / (number_of_nodes * (number_of_nodes - 1))))
        print('Average-degree density: ' + str(number_of_edges / number_of_nodes))

        for layer in multilayer_graph.layers_iterator:
            self.print_layer_info(multilayer_graph, layer)


    def print_layer_info(self, multilayer_graph, layer):
        number_of_nodes = multilayer_graph.number_of_nodes
        number_of_edges = multilayer_graph.get_number_of_edges(layer)

        print('------------- Layer ' + str(multilayer_graph.get_layer_mapping(layer)) + ' -------------')
        print('Edges: ' + str(number_of_edges))
        print('Edge density: ' + str((2 * number_of_edges) / (number_of_nodes * (number_of_nodes - 1))))
        print('Average-degree density: ' + str(number_of_edges / number_of_nodes))


    def print_end_algorithm(self):
        print('Execution Time: ' + str(self.time) + 's')
        print('Preprocess Memory Usage: ' + str(self.preprocess_memory) + 'MB')
        print('Algorithm Memory Usage: ' + str(memory_usage_resource() - self.preprocess_memory) + 'MB')


    def print_densest_subgraph(self, beta, maximum_density, densest_subgraph, maximum_layers, densest_subgraph_core, maximum_average_degrees):

        print('Beta: ' + str(beta))
        print('Rho: ' + str(maximum_density))
        print('Size: ' + str(densest_subgraph))
        print('Number of selected layers: ' + str(len(maximum_layers)))
        print('Selected layers: ' + str([layer + 1 for layer in maximum_layers]).replace('[', '').replace(']', ''))
        print('Core number: ' + str(densest_subgraph_core))
        print('Average-degree density vector: ' + str(tuple(round(average_degree, 2) for average_degree in maximum_average_degrees)))


    @staticmethod
    def current_milliseconds():
        return int(round(time() * 1000))