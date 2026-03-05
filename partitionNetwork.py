import sys, os
import partitionNetworkHelper, partitionComputations
import libsbml
import networkx as nx
from copy import deepcopy
from tqdm import tqdm
import time
import pickle

def readArguments():
    '''Reading arguments 

        Upon invocation this function reads the arguments that were passed from the commandline.
        
        Parameters
        ----------
        None

        Returns:
        - str: inputFilePath - the path directing to the input sbml file
        - int: maxThreads - number of maximum threads that can be used for parallelization
        - int: cutLength - the minimum length subnetwork in the generated R-Graph (the smallest partitions)
        - str: outputPickleFiles - path specifying where the pickle-files with the data on the partitioned network are going to be stored
        - bool: smallMoleculesBool - specifying if small highly intersecting molecules should be excluded from the network or not
        - str: smallMoleculePath - path to the file specifying small molecules that should be exluded from the network
        '''
    
    inputBool = False
    threadBool = False
    cutOffBool = False
    outputBool = False
    smallMoleculesBool = False
    for k in range(len(sys.argv)):
        newArgument = sys.argv[k]
        if newArgument == "-i" or newArgument == "--input":
            inputFilePath = sys.argv[k+1]
            inputBool = True
        if newArgument == "-t" or newArgument == "--threads":
            maxThreads = int(sys.argv[k+1])
            threadBool = True
        if newArgument == "-c" or newArgument == "--cutOff":
            cutOff = int(sys.argv[k+1])            
            cutOffBool = True
        if newArgument == "-o" or newArgument == "--ouput":
            outputPickleFiles=sys.argv[k+1]
            outputBool = True
        if newArgument == "-s" or newArgument == "--smallMolecules":
            smallMoleculePath = sys.argv[k+1]
            smallMoleculesBool = True
    if inputBool == False:
        sys.exit("Please provide an input file.")
    if threadBool==False:
        maxThreads = 1
    if cutOffBool == False:
        cutOff = 2
    if outputBool == False:
        if os.path.exists("./Output") == False:
            os.makedirs("./Output")
        outputPickleFiles = "./Output"
    if smallMoleculesBool == False:
        smallMoleculePath = ""
    return inputFilePath, maxThreads, cutOff, outputBool, outputPickleFiles, smallMoleculesBool, smallMoleculePath
########################################
########################################

def readSmallMolecules(smallMoleculesPath:str):
    ''' Read small Molecules
    
    Upon invocation this function reads a set of small and from the perspective of the user unneccessary
    molecules that are being removed from the network.
    
    Parameters:
    ----------
    
    smallMoleculesPath : str
        Specifies the path to a file wit the molecules that should be removed from the network and not considered 
        for analysis 
    
    Returns:
    - set: A set of strings specifying small molecules exluded from the analyzed network.
    '''
    smallMolecules = set()
    with open(smallMoleculesPath, "r") as file:
        while True:
            line = file.readline().strip()
            if line=="":
                break
            else:
                shortendNode = "_".join(line.split("_")[:-1])+"_"
                smallMolecules.add(shortendNode)
    return smallMolecules
########################################
########################################

#=============================================================================#
#                                   Main                                      #
#=============================================================================#

# 0. Read input parameters
timeStamp = time.time()
inputFilePath, maxThreads, cutOff, outputBool, outputPickleFiles, smallMoleculesBool, smallMoleculesPath = readArguments()

# 1. Define variables
parameters = {}
parameters["path"] = inputFilePath

# 2. Read Model
reader = libsbml.SBMLReader()
document = reader.readSBML(inputFilePath)
model = document.getModel()

# 3. Read small molecules
if smallMoleculesBool == False:
    smallMolecules = set() 
else:
    smallMolecules = readSmallMolecules(smallMoleculesPath)

# 4. Build network
metabolicNetwork, vertexIDs = partitionNetworkHelper.buildNetwork(model=model)
inhibitors = {}      # not yet implemented
                       
# 4.1 Find/Determine/Define abundant molecules to inhibit to many crosslinkings between modules that are actually distant from each other
unnecessaryMolecules = partitionNetworkHelper.getAbundantMolecules(smallMolecules, metabolicNetwork)
usefulNetwork = deepcopy(metabolicNetwork)                              
nodes = deepcopy(set(metabolicNetwork.nodes()))

unnecessaryMoleculesIDSet = set()
for m in unnecessaryMolecules:
    if m in vertexIDs:
        unnecessaryMoleculesIDSet.add(vertexIDs[m])
parameters["unnecessaryMolecules"] = unnecessaryMolecules
parameters["unnecessaryMoleculesIDs"] = unnecessaryMoleculesIDSet

metabolicNetwork.remove_nodes_from(unnecessaryMoleculesIDSet)     # Network for constructing the submodules
usefulNetwork.remove_nodes_from(unnecessaryMoleculesIDSet)        # Network that will be analyzed in the end for elementary circuits

# 5. Partitioning and Analysis 
wCCList = list(nx.weakly_connected_components(metabolicNetwork))
treeCounter = 0
parameters["metabolicNetwork"]=deepcopy(metabolicNetwork)
parameters["usefulNetwork"]=usefulNetwork
parameters["nodes"]=usefulNetwork.nodes()
parameters["vertexIDs"] = vertexIDs
for i in tqdm(range(len(wCCList)), desc="Weakly connected components"):
    cSet = wCCList[i]
    if len(cSet)>1:
        connectedComponent = nx.subgraph(metabolicNetwork, cSet).copy()
        for scSet in nx.strongly_connected_components(connectedComponent):                      # Only analyze strongly connected components
            stronglyConnectedComponent = nx.subgraph(connectedComponent, scSet).copy()
            X, Y = nx.bipartite.sets(stronglyConnectedComponent)
            if len(X)>0:
                parameters["reactions"], parameters["metabolites"] = partitionNetworkHelper.getReactionsAndMetabolites(X,Y, metabolicNetwork)
                print("Reactions:", len(parameters["reactions"]), "Metabolites:", len(parameters["metabolites"]))
                reactionNetwork = partitionNetworkHelper.createReactionNetwork(stronglyConnectedComponent, parameters["reactions"], inhibitors)
                if len(reactionNetwork)<cutOff:
                    print("Exiting because reaction network is too small")
                    continue

                # Define new variables that are necessary
                if len(reactionNetwork)>100:
                    noThreads = maxThreads
                else:
                    noThreads = 2
                partitionTree = nx.DiGraph()
                siblings = {}
                leaves = set()
                uRN = reactionNetwork.to_undirected(reactionNetwork, as_view=False) 
                partitionTree.add_node(uRN)                           # partition tree has only undirected graphs as nodes
                s, Q, nodes = partitionComputations.computePartitioning(reactionNetwork)
                if Q<=0:                                              # If Q == 0 or smaller don't partition
                    leaves.add(uRN)
                    continue
                partitionComputations.continuePartitioning(s, nodes, uRN, partitionTree, cutOff, siblings, leaves, reactionNetwork, noThreads)

                partitionTreePath = "./PickleFiles/" + outputPickleFiles+ "/partitionTree"+str(treeCounter) + ".pkl"
                if not os.path.exists("./PickleFiles"):
                    os.makedirs("./PickleFiles")
                if not os.path.exists("./PickleFiles/"+outputPickleFiles):
                    os.makedirs("./PickleFiles/"+outputPickleFiles)
                
                with open(partitionTreePath, "wb") as file:
                    pickle.dump((parameters, partitionTree, siblings, leaves, uRN, usefulNetwork), file)
                    treeCounter +=1
                    file.close()

