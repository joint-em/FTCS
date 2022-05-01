from FirmCore.FirmCore import FirmCore, FirmCore_decomposition, FirmCore_k
from FirmTruss.FirmTruss import FirmTruss, FirmTruss_decomposition, FirmTruss_k
from MLGraph.multilayer_graph import MultilayerGraph
from Utils.memory_measure import memory_usage_resource
from Utils.information import Information
from time import time
import argparse


if __name__ == '__main__':
    # create a parser
    parser = argparse.ArgumentParser(description='Firm Structures in Multilayer Networks')

    # arguments
    parser.add_argument('d', help='dataset')
    parser.add_argument('m', help='method')
    parser.add_argument('a', default='', help='algorithm')
    # options
    parser.add_argument('--dic', dest='dic', action='store_true', default=True ,help='dicomposition')
    parser.add_argument('--save', dest='save', action='store_true', default=False ,help='save results')

    parser.add_argument('-k', help='k', type=int)
    parser.add_argument('-l', help='lambda', type=int)
    parser.add_argument('-q', help='query nodes', type=list)

    # read the arguments
    args = parser.parse_args()

    # dataset path
    dataset_path = "/Code/Datasets/" + args.d

    information = Information(args.d)

    information.print_dataset_name(dataset_path)

    # create the input graph and print its name
    start = int(round(time() * 1000))
    
    multilayer_graph = MultilayerGraph(dataset_path)
    
    end = int(round(time() * 1000))
    print(" >>>> Preprocessing Time: ", (end - start)/1000.00, " (s)\n")
    
    # FirmCore decomposition algorithms
    if args.m == 'core':
        if args.dic:
            print('---------- FirmCore Dicomposition ----------')
            FirmCore_decomposition(multilayer_graph.adjacency_list, multilayer_graph.nodes_iterator, multilayer_graph.layers_iterator, information, save=args.save)
        else:
            print('---------- FirmCore lambda = %d ----------'%args.l)
            FirmCore(multilayer_graph.adjacency_list, multilayer_graph.nodes_iterator, multilayer_graph.layers_iterator, args.l, information, save=args.save)

       
    # FirmTruss decomposition algorithms
    elif args.m == 'truss':
        if args.dic:
            print('---------- FirmTruss Dicomposition ----------')
            FirmTruss_decomposition(multilayer_graph.adjacency_list, multilayer_graph.nodes_iterator, multilayer_graph.layers_iterator, information, save=args.save)
        else:
            print('---------- FirmTruss lambda = %d ----------'%args.l)
            FirmCore(multilayer_graph.adjacency_list, multilayer_graph.nodes_iterator, multilayer_graph.layers_iterator, args.l, information, save=args.save)
            
    
           
    # Community algorithms
    elif args.m == 'community':
        if args.a == 'FirmTruss-Global':
            print('---------- FirmTruss Global Algorithm ----------')
            FirmTruss_k(multilayer_graph.adjacency_list, multilayer_graph.nodes_iterator, multilayer_graph.layers_iterator, args.k, args.l, args.q, information, save=args.save)
        # else:
        #     print('---------- FirmTruss lambda = %d ----------'%args.l)
        #     FirmCore(multilayer_graph.adjacency_list, multilayer_graph.nodes_iterator, multilayer_graph.layers_iterator, args.l, information, save=args.save)
            


    # dataset information
    elif args.m == 'info':
        information.print_dataset_info(multilayer_graph)

    # unknown input
    else:
        parser.print_help()









