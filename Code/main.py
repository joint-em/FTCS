from FirmTruss.FirmTruss import FirmTruss_Local, FirmTruss_decomposition, FirmTruss_Global, FirmTruss_k
from MLGraph.multilayer_graph import MultilayerGraph
from Utils.memory_measure import memory_usage_resource
from Utils.information import Information
from time import time
from Homophily.Homophily import AFTCS_Approx
import argparse


if __name__ == '__main__':
    # create a parser
    parser = argparse.ArgumentParser(description='FirmTruss-based Community Search in Multilayer Networks')

    # arguments
    parser.add_argument('d', help='dataset')
    parser.add_argument('m', help='algorithms {FirmTruss, Global, iGlobal, Local, iLocal, AFTCS-Approx}')
    # parser.add_argument('a', default='', help='algorithms {FirmTruss-Global, FirmTruss-Local, AFTCS-Approx}')

    # options
    # parser.add_argument('--dic', dest='dic', action='store_true', default=True ,help='dicomposition')
    parser.add_argument('--save', dest='save', action='store_true', default=False ,help='save results')

    parser.add_argument('-k', help='k', type=int)
    parser.add_argument('-p', help='p', type=float)
    parser.add_argument('-l', help='lambda', type=int)
    parser.add_argument('-q', help='query nodes', type=int)

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
    
    # FirmTruss decomposition algorithms
    if args.m == 'FirmTruss':
        print('---------- FirmTruss Dicomposition ----------')
        FirmTruss_decomposition(multilayer_graph.adjacency_list, multilayer_graph.nodes_iterator, multilayer_graph.layers_iterator, information, save=args.save)
       
    # Community algorithms
    elif args.m == 'Global':
        print('---------- FirmTruss Global Algorithm ----------')
        FirmTruss_Global(multilayer_graph.adjacency_list, multilayer_graph.nodes_iterator, multilayer_graph.layers_iterator, args.k, args.l, args.q, information, save=args.save, index=False)

    elif args.m == 'iGlobal':
        print('---------- FirmTruss Index-based Global Algorithm ----------')
        FirmTruss_Global(multilayer_graph.adjacency_list, multilayer_graph.nodes_iterator, multilayer_graph.layers_iterator, args.k, args.l, args.q, information, save=args.save, index=True)
    
    elif args.m == 'Local':
        print('---------- FirmTruss Local Algorithm ----------')
        FirmTruss_Local(multilayer_graph.adjacency_list, multilayer_graph.nodes_iterator, multilayer_graph.layers_iterator, args.k, args.l, args.q, information, save=args.save)
    
    elif args.m == 'iLocal':
        print('---------- FirmTruss iLocal Algorithm ----------')
        FirmTruss_Local(multilayer_graph.adjacency_list, multilayer_graph.nodes_iterator, multilayer_graph.layers_iterator, args.k, args.l, args.q, information, save=args.save)
    
    elif args.m == 'AFTCS-Approx':
        attribute_path = "/Code/Datasets/" + args.d + "_attributes"
        multilayer_graph.load_attributes(attribute_path)
        print('---------- AFTCS-Approx Algorithm ----------')
        AFTCS_Approx(multilayer_graph.adjacency_list, multilayer_graph.attributes, multilayer_graph.nodes_iterator, multilayer_graph.layers_iterator, args.k, args.l, args.p, args.q, information, save=args.save)
        


    # dataset information
    elif args.m == 'info':
        information.print_dataset_info(multilayer_graph)

    # unknown input
    else:
        parser.print_help()









