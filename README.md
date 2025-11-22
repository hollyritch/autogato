# 1. General

Autogato is a python package designed to compute irreducible autocatalytic subsystems in metabolic networks. It consists of the following modules:

- partitionNetwork.py
- partitionNetworkHelper.py
- partitionComputations.py
- partitionNetwork.py
- partitionAnalysis.py
- setup.py
- checkMatch.pyx

It takes sbml-models as input, currently especially BIGG-Models. Code will be adapated to accept also other models. 

# 2. Installation and Setup

Prior to usage, setup a new conda environment with Python 3 (presumably >=3.12) and install the following packages:

- SymPy
- NumPy
- SciPy
- NetowrkX
- libsbml
- Concurrent.Futures
- tqdm
- itertools
- Cython

Clone the git, activate your conda environment, and first perform the following command:

python setup.py build_ext --inplace

to compile checkMatch.pyx so that it can be imported as a normal in all partitionAnalysis.py

# 3. Usage

3.1 Partition your Network

First call partition Network via the following command:

- python partitionNetwork.py -i 
