import sys, os
import partitionNetworkHelper, partitionComputations
import libsbml
import networkx as nx
from copy import deepcopy
from tqdm import tqdm
import time
import pickle

def readArguments():
    inputBool = False
    threadBool = False
    cutOffBool = False
    outputBool = False
    smallMoleculesBool = False
    for k in range(sys.argv()):
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
    if smallMoleculesBool == True:
        smallMoleculePath = ""
    return inputFilePath, maxThreads, cutOff, outputBool, outputPickleFiles, smallMoleculesBool, smallMoleculePath
########################################
########################################

def readSmallMolecules(smallMoleculesPath):
    smallMolecules = set()
    with open(smallMoleculesPath, "r") as file:
        while True:
            line = file.readline().strip()
            if line=="":
                break
            else:
                shortendNode = "_".join(line.split("_")[:-1])+"_"
                if shortendNode in exclude:
                    print("Excluding",line)
                else:
                    smallMolecules.add(shortendNode)
    return smallMolecules
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

# 4.1 Find/Determine/Define abundant molecules to inhibit to many crosslinkings between modules that are actually distant from each other

if outputPickleFiles=="EColiCore":
    abundantMolecules = {"M_adp_c", "M_h_c", "M_coa_c", "M_h2o_c", "M_nad_c", "M_nadh_c", "M_h_e", "M_pi_e", "M_pi_c", "M_co2_c", "M_amp_c", "M_co2_e", "M_o2_c", "M_q8h2_c", "M_q8_c", "M_nadp_c", "M_nadph_c", "M_h2o_e", "M_nh4_e", "M_o2_e", "M_nh4_c"}
    unnecessaryMolecules = {"M_adp_c", "M_h2o_c", "M_co2_c", "M_h_c", "M_coa_c", "M_nad_c", "M_nadh_c", "M_h_e", "M_pi_e", "M_pi_c", "M_amp_c", "M_co2_e", "M_o2_c", "M_q8h2_c", "M_q8_c", "M_nadp_c", "M_nadph_c", "M_h2o_e", "M_nh4_e", "M_o2_e", "M_nh4_c"}        

    # 3.2 Get inhibitors if necesary
    
    unnecessaryMolecules, abundantMolecules = partitionNetworkHelper.getAbundantMolecules(smallMolecules, metabolicNetwork, {})

unnecessaryMoleculesIDSet = set()
#partitionNetworkHelper.plotDegreeDistribution(metabolicNetwork)
usefulNetwork = deepcopy(metabolicNetwork)                              
highest = 0
nodes = deepcopy(set(metabolicNetwork.nodes()))
parameters["unnecessaryMolecules"] = unnecessaryMolecules
parameters["abundantMolecules"] = abundantMolecules
parameters["unnecessaryMolecules"] = unnecessaryMoleculesIDSet
for m in unnecessaryMolecules:
    if m in vertexIDs:
        unnecessaryMoleculesIDSet.add(vertexIDs[m])

# 3.4 Remove unnecessary stuf‚
metabolicNetwork.remove_nodes_from(unnecessaryMoleculesIDSet)                  # Network for constructing the submodules
usefulNetwork.remove_nodes_from(unnecessaryMoleculesIDSet)                   # Network that will be analyzed in the end for elementary circuits

# TODO: Problem: This has the disadvantage that reactions might get lost.
#metabolicNetwork.remove_nodes_from(abundantMolecules)                  # Network for constructing the submodules

#metabolicNetwork = buildExampleNetwork()                               # Only for toy example
#partitionNetworkHelper.plotDegreeDistribution(metabolicNetwork)
# 4. Partitioning and Analysis 
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
        for scSet in nx.strongly_connected_components(connectedComponent):
            stronglyConnectedComponent = nx.subgraph(connectedComponent, scSet).copy()
            X, Y = nx.bipartite.sets(stronglyConnectedComponent)
            if len(X)>0:
                parameters["reactions"], parameters["metabolites"] = partitionNetworkHelper.getReactionsAndMetabolites(X,Y, metabolicNetwork)
                print("Reactions:", len(parameters["reactions"]), "Metabolites:", len(parameters["metabolites"]))
                # print("Metabolites:", parameters["metabolites"])
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
                uRN = partitionNetworkHelper.generateUndirectedReactionNetwork(reactionNetwork)
                partitionTree.add_node(uRN)
                s, Q, nodes = partitionComputations.computePartitioning(reactionNetwork, noThreads)
                if Q<=0:                                                                    # If Q == 0 or below don't partition
                    leaves.add(uRN)
                    continue
                partitionComputations.continuePartitioning(s, nodes, uRN, partitionTree, cutOff, siblings, leaves, reactionNetwork, noThreads)
                partitionNetworkHelper.performSanityChecks(leaves, partitionTree, siblings)
                maxOverlap, overlapDict, metaboliteOverlapDict = partitionNetworkHelper.analyseOverlap(partitionTree, metabolicNetwork, siblings)

                partitionTreePath = "./PickleFiles/" + outputPickleFiles+ "/partitionTree"+str(treeCounter) + ".pkl"
                if not os.path.exists("./PickleFiles"):
                    os.makedirs("./PickleFiles")
                if not os.path.exists("./PickleFiles/"+outputPickleFiles):
                    os.makedirs("./PickleFiles/"+outputPickleFiles)
                
                with open(partitionTreePath, "wb") as file:
                    pickle.dump((parameters, partitionTree, siblings, leaves, uRN, usefulNetwork), file)
                    treeCounter +=1
                    file.close()

