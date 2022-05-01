FirmTruss Community Search in Multilayer Network: Implementations
================================================

This folder contains all code and datasets that are used in "FirmTruss Community
Search in Multilayer Network".

 

Jupyter Notebook
----------------

We provide a jupyter notebook ("Sample_of_Experiments.ipynb") for convenience,
including many examples and samples of experiments. Please note that due to the storage limit on GitHub, indices for some datasets are not provided. If you receive an error about missing a file, please run the following code for the dataset (replace X with the name of the dataset):

 python main.py X FirmTruss —save
 
 This command will save FirmTruss indices and will solve the issue.



Folders
-------

Datasets: Include all used datasets

FirmCore: Implementation of FirmCore structure (which is used in FirmTruss as
well)

FirmTruss: Implementation of FirrmTruss decomposition, Finding_G_0, Global,
iGlobal, Local, and iLocal algorithms

Homophily: Implementation of AFTCS-Approx, and Exact_MaxMin algorithms

MLGraph: Implementation of multilayer graphs

Scripts: Implementation of needed functions to reproduce the experiment's
results.

Setup: Needed files for compiling the code with Cython

output: Includes all indices for each dataset

 

Code
----

If you have Cython available, first run 'python setup.py build_ext --inplace'
from the folder 'Code/'. This command builds the .c files created by Cython.
Alternatively, without running the mentioned command, it is still possible to
directly execute the Python code.

 

Execution
---------

Run the following command from the folder "Code/"

" python main.py [-h] [--save] [-k K] [-p P] [-l L] [-q Q] d m "

positional arguments: d dataset m algorithms {FirmTruss, Global, iGlobal, Local,
iLocal, AFTCS-Approx}

optional arguments: -h, --help show this help message and exit --save save
results -k K k -p P p -l L lambda -q Q query nodes

**Any value of p smaller than -100 (resp. more than +100), is considered as
-infty (resp. + infty)**

\#\#\# Examples

1.  python main.py td iGlobal -k 12 -l 10 -q 77

2.  python main.py dblp Local -k 4 -l 1 -q 63

3.  python main.py terrorist AFTCS-Approx -k 5 -l 2 -q 4 -p -200

4.  python main.py RM AFTCS-Approx -k 23 -l 2 -q 4 -p 1

 

\*\*Note\*\*

1. If for dataset X, indices are not available, you can run the following command:

python main.py X FirmTruss —save


2. Due to the storage limit on Github, some large datasets are not avilable in Dataset folder. To download large datasets please use the following link:


