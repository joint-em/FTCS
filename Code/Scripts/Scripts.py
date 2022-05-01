from FirmTruss.FirmTruss import FirmTruss_Local, FirmTruss_decomposition, FirmTruss_Global
from MLGraph.multilayer_graph import MultilayerGraph
from Utils.density import density
from FirmCore.FirmCore import FirmCore_Global
from Homophily.Homophily import AFTCS_Approx
from Utils.information import Information
from tqdm import tqdm
from time import time
import statistics
import pickle
import random
import time
import os



def density_of_community(dataset, query, attributes=False, p=None):
    ML_graph = MultilayerGraph('/Code/Datasets/' + dataset)
    information = Information(dataset)
    if attributes:
        attribute_path = "../Datasets/" + dataset + "_attributes"
        ML_graph.load_attributes(attribute_path)


    indices = [[]]

    for lammy in range(1, len(ML_graph.layers_iterator) + 1):
        index_file = open("./output/" + information.dataset + "_" + str(lammy) + "_FirmTruss_decomposition.pkl", "rb")
        index = pickle.load(index_file)
        indices.append(index)   
    
    sft = []

    for lammy in range(1, len(ML_graph.layers_iterator) + 1):
        max_ind = 0
        for edgeS in indices[lammy].keys():
            if edgeS[0] == query or edgeS[1] == query:
                if max_ind < indices[lammy][edgeS]:
                    max_ind = indices[lammy][edgeS]

        sft.append((max_ind, lammy))
    
    sft1= (0, 0)
    counter = 0
    while sft1[0] < 4 or sft1[1] < 1:
        sft1 = random.choices(sft, k=1)[0]
        counter += 1
        if counter > 100:
            break

    k = sft1[0]
    lammy = sft1[1]
    
    if attributes:
        Community = AFTCS_Approx(ML_graph.adjacency_list, 
                        ML_graph.attributes,
                                        ML_graph.nodes_iterator, 
                                        ML_graph.layers_iterator, 
                                        k, 
                                        lammy,  
                                        p,
                                        query,
                                        information,   
                                        verbos=False)

    else:    
        Community = FirmTruss_Global(ML_graph.adjacency_list, 
                                            ML_graph.nodes_iterator, 
                                            ML_graph.layers_iterator, 
                                            k, 
                                            lammy,  
                                            query,
                                            information,   
                                            verbos=False)



    number_of_edges = {}

    for layer in ML_graph.layers_iterator:
        number_of_edges[layer] = 0

    for node in Community:
        for layer, layer_neighbors in enumerate(ML_graph.adjacency_list[node]):
            for neighbour in set(layer_neighbors) & set(Community):
                if node > neighbour:
                    number_of_edges[layer] += 1


    density_value = density(len(Community), number_of_edges, 1)[0]

    print(">>>>>> Density of this community is :", density_value)








def community_evaluation(dataset, C, query, attributes=False, p=None):
    ML_graph = MultilayerGraph('/Code/Datasets/' + dataset)
    information = Information(dataset)
    if attributes:
        attribute_path = "/Code/Datasets/" + dataset + "_attributes"
        ML_graph.load_attributes(attribute_path)

    indices = [[]]

    for lammy in range(1, len(ML_graph.layers_iterator)):
        index_file = open("./output/" + information.dataset + "_" + str(lammy) + "_FirmTruss_decomposition.pkl", "rb")
        index = pickle.load(index_file)
        indices.append(index)

    sft = dict({})

    for node in ML_graph.nodes_iterator:
        sft[node] = [0]
        for lammy in range(1, len(ML_graph.layers_iterator)):
            max_ind = 0
            for edgeS in indices[lammy].keys():
                if edgeS[0] == node or edgeS[1] == node:
                    if max_ind < indices[lammy][edgeS]:
                        max_ind = indices[lammy][edgeS]
            sft[node].append(max_ind)


    best_f1 = 0
    for lammy in range(1, len(ML_graph.layers_iterator)):
        f1 = 0
        try:
            k = sft[query][lammy]

            if attributes:
                community = AFTCS_Approx(ML_graph.adjacency_list,
                                            ML_graph.attributes,
                                            ML_graph.nodes_iterator, 
                                            ML_graph.layers_iterator, 
                                            k, 
                                            lammy, 
                                            p,
                                            query, 
                                            information = information,
                                            verbos=False)


            community = FirmTruss_Global(ML_graph.adjacency_list, 
                                        ML_graph.nodes_iterator, 
                                        ML_graph.layers_iterator, 
                                        k, 
                                        lammy, 
                                        query, 
                                        information, 
                                        verbos=False)

            for i in range(len(C)):
                if query in C[i]:
                    pre = len(C[i] & community) / len(C[i])
                    recall = len(C[i] & community) / len(community)
                    f1 = 2 * pre * recall / (pre + recall)

                if f1 > best_f1:
                    best_f1 = f1
        except:
            best_f1 = 1

    
    print(">>>>>> F1-score :", best_f1)





# Brain Networks


def creat_dataset(n_sample, _type):
    files = os.listdir('./ADHD/graphs/' + _type)
    files = random.choices(files, k=n_sample)
    
    edge = set()
    layer = 0
    for ffile in files:
        layer += 1
        file_add = "./ADHD/graphs/" + _type + '/' + ffile

        with open(file_add) as f:
            lines = f.readlines()
            counter = 0
            for line in lines:
                counter += 1
                neighbours = line.split(" ")
                for i in range(len(neighbours)):
                    if int(neighbours[i]) == 1:
                        edge.add((min(counter, i+1), max(counter, i+1), layer))
    
    counter = 0
    with open('./Datasets/Temp_datasets/' + _type + '_train/data.txt', 'w') as f:
        output = str(n_sample) + " " + str(190) + " " + str(190)
        f.write(output)
        f.write('\n')
        for e in edge:
            counter += 1
            output = str(e[2]) + " " + str(e[0]) + " " + str(e[1])
            f.write(output)
            f.write('\n')


def creat_test_dataset(n_sample, _type):
    files = os.listdir('./ADHD/graphs/' + _type)
    files = random.choices(files, k=n_sample)
    
    sample = -1
    for ffile in files:
        sample += 1
        edge = set()
        layer = 1
        file_add = "./ADHD/graphs/" + _type + '/' + ffile

        with open(file_add) as f:
            lines = f.readlines()
            counter = 0
            for line in lines:
                counter += 1
                neighbours = line.split(" ")
                for i in range(len(neighbours)):
                    if int(neighbours[i]) == 1:
                        edge.add((min(counter, i+1), max(counter, i+1), layer))
    
        counter = 0
        with open('./Datasets/Temp_datasets/' + _type + '_test/data' + str(sample) + '.txt', 'w') as f:
            output = str(n_sample) + " " + str(190) + " " + str(190)
            f.write(output)
            f.write('\n')
            for e in edge:
                counter += 1
                output = str(e[2]) + " " + str(e[0]) + " " + str(e[1])
                f.write(output)
                f.write('\n')


def similarity(C_1, C_2, measure="F1"):
    if measure == "F1":
        a = len(set(C_1) & set(C_2)) / len(set(C_2))
        b = len(set(C_1) & set(C_2)) / len(set(C_1))

        return 2.000 * (a * b) / (a + b)

    elif measure == "overlap":
        return float(len(set(C_1) & set(C_2))/min(len(C_1), len(C_2)))
    
    elif measure == "Jaccard":
        return len(set(C_1) & set(C_2))/len(set(C_1).union(set(C_2)))
    
    else:
        return 0


def ADHD_classification(n_adhd, n_td, k_td, lammy_td, k_adhd, lammy_adhd, query_node, measure="overlap", creat_data=True, verbos=True):
    
    # creat test and training data
    if creat_data:
        # td
        creat_dataset(n_td, 'td')
        creat_test_dataset(330 - n_td, 'td')
        
        # adhd
        creat_dataset(n_adhd, 'adhd')
        creat_test_dataset(190 - n_adhd, 'adhd')
    
    
    multilayer_graph_td = MultilayerGraph('/Code/Datasets/Temp_datasets/td_train/data')
    multilayer_graph_adhd = MultilayerGraph('/Code/Datasets/Temp_datasets/adhd_train/data')
    
    information = Information("TD")

    community_td = FirmTruss_Local(multilayer_graph_td.adjacency_list, multilayer_graph_td.nodes_iterator, multilayer_graph_td.layers_iterator, k_td, lammy_td, query_node, information, verbos=False)
    
    if verbos:
        print("Community TD: ", community_td)
    
    information = Information("ADHD")

    community_adhd = FirmTruss_Local(multilayer_graph_adhd.adjacency_list, multilayer_graph_adhd.nodes_iterator, multilayer_graph_adhd.layers_iterator, k_adhd, lammy_adhd, query_node, information, verbos=False)

    if verbos:
        print("Community ADHD: ", community_adhd)
    
    time.sleep(1)
    
    error_adhd = 0
    error_td = 0
    all_td = 330 - n_td
    all_adhd = 190 - n_adhd
    
    for i in tqdm(range(330 - n_td)):
        information = Information("sample")
        
        graph_sample = MultilayerGraph('/Code/Datasets/Temp_datasets/td_test/data' + str(i))
        
        sample_community = None
        t = - 3
        
        while sample_community is None:
            sample_community = FirmTruss_Local(graph_sample.adjacency_list, graph_sample.nodes_iterator, graph_sample.layers_iterator, k_td - t, 1, query_node, information, verbos=False)
            t += 1
            if t > 3:
                break
            
        if sample_community is not None:
            similarity_score_td = similarity(sample_community, community_td, measure)
            similarity_score_adhd = similarity(sample_community, community_adhd, measure)
            
            if similarity_score_adhd > similarity_score_td:
                error_td += 1
        
        else:
            all_td -= 1
    
    if verbos:
        print("Accuracy in TD: ", 1.0000 - error_td/(all_td))
    
    time.sleep(1)
    
    for i in tqdm(range(190 - n_adhd)):
        information = Information("sample")
        
        graph_sample = MultilayerGraph('/Code/Datasets/Temp_datasets/adhd_test/data' + str(i))

        
        sample_community = None
        t = - 3
        
        while sample_community is None:
            sample_community = FirmTruss_Local(graph_sample.adjacency_list, graph_sample.nodes_iterator, graph_sample.layers_iterator, k_adhd - t, 1, query_node, information, verbos=False)
            t += 1
            if t > 3:
                break
        
        if sample_community is not None:
            similarity_score_td = similarity(sample_community, community_td, measure)
            similarity_score_adhd = similarity(sample_community, community_adhd, measure)
        
            if similarity_score_adhd < similarity_score_td:
                error_adhd += 1
        
        else:
            all_adhd -= 1
    if verbos:
        print("Accuracy in ADHD: ", 1.0000 - error_adhd/(all_adhd))
    
    precision = (all_adhd - error_adhd)/((all_adhd - error_adhd) + error_td)
    
    recall = (all_adhd - error_adhd) / (all_adhd)
    
    accuracy = ((all_adhd - error_adhd) + (all_td - error_td)) / (all_adhd + all_td)
    
    f1 = 2*precision*recall / (precision + recall)
    
    if verbos:
        print("Precision: ", precision, ", Recall: ", recall, ", Accuracy: ", accuracy, ", F1 Score: ", f1)
    
    return precision, recall, accuracy, f1
    