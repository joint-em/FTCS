FirmTruss Community Search in Multilayer Network
================================================

This repository contains the implementation of algorithms, used datasets, and the full version of paper "FirmTruss Community Search in Multilayer Network" (VLDB 2023). 


#### Authors: [Ali Behrouz](https://abehrouz.github.io/), [Farnoosh Hashemi](https://farnooshha.github.io//), [Laks V.S. Lakshmanan](https://www.cs.ubc.ca/~laks/)
#### [Link to the paper](https://) ([Arxiv](https://arxiv.org/pdf/2205.00742.pdf))
#### [Poster]()
#### [Brief video explanation]()





### Key Contributions
----------------  
1. We extend the notion of Trusses to multilayer networks with polynomial time decomposition. We then theoretically guarantee that it has a bounded diameter, high density, and high edge connectivity.
2. We present a new community model based on FirmTruss, called FTCS, that provably has a small diameter.
3. We propose a new community model based on network homophily that maximizes the p-mean of the community homophily score.
4. We suggest brain network analysis as a new application for the problem of community search (both in multilayer and single-layer networks).




### Abstract
----------------  
In applications such as biological, social, and transportation networks, interactions between objects span multiple aspects. For accurately modeling such applications, multilayer networks have been proposed. Community search allows for personalized community discovery and has a wide range of applications in large real-world networks. While community search has been widely explored for single-layer graphs, the problem for multilayer graphs has just recently attracted attention. Existing community models in multilayer graphs have several limitations, including disconnectivity, free-rider effect, resolution limits, and inefficiency. To address these limitations, we study the problem of community search over large multilayer graphs. We first introduce FirmTruss, a novel dense structure in multilayer networks, which extends the notion of truss to multilayer graphs. We show that FirmTrusses possess nice structural and computational properties and bring many advantages compared to the existing models. Building on this, we present a new community model based on FirmTruss, called FTCS, and show that finding an FTCS community is NP-hard. We propose two efficient 2-approximation algorithms, and show that no polynomial-time algorithm can have a better approximation guarantee unless P = NP. We propose an index-based method to further improve the efficiency of the algorithms. We then consider attributed multilayer networks and propose a new community model based on network homophily. We show that community search in attributed multilayer graphs is NP-hard and present an effective and efficient approximation algorithm. Experimental studies on real-world graphs with ground-truth communities validate the quality of the solutions we obtain and the efficiency of the proposed algorithms.




### Code
----------------  
This folder includes the implementation of all algorithms, datasets, and a Jupyter notebook for convenience, including many examples and samples of experiments. For more details, please refer to the README file in this folder.  





### Reference
----------------  
```
@misc{FTCS2022,
  author = {Behrouz, Ali and Hashemi, Farnoosh and Lakshmanan, Laks V. S.},
  title = {FirmTruss Community Search in Multilayer Networks},  
  doi = {10.48550/ARXIV.2205.00742},
  url = {https://arxiv.org/abs/2205.00742},
  publisher = {arXiv},
  year = {2022}
}
```
