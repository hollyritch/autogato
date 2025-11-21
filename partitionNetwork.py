import sys, os
import partitionNetworkHelper, partitionComputations
import libsbml
import networkx as nx
from copy import deepcopy
from tqdm import tqdm
import time
import pickle


########################################
########################################
# Main
# 0. Read input parameters
timeStamp = time.time()
inputFilePath = sys.argv[1]
cutOff = int(sys.argv[2])
maxThreads = int(sys.argv[3])
outputPickleFiles = sys.argv[4]
# 1. Define variables
parameters = {}
parameters["path"] = inputFilePath

# 2. Read Model
reader = libsbml.SBMLReader()
document = reader.readSBML(inputFilePath)
model = document.getModel()

#parameters["model"] = model

# 3. Build network
metabolicNetwork, vertexIDs = partitionNetworkHelper.buildNetwork(model=model)
inhibitors ={}
# 3.1 Find/Determine/Define abundant molecules to inhibit to many crosslinkings between modules that are actually distant from each other
if outputPickleFiles=="EColiCore":
    abundantMolecules = {"M_adp_c", "M_h_c", "M_coa_c", "M_h2o_c", "M_nad_c", "M_nadh_c", "M_h_e", "M_pi_e", "M_pi_c", "M_co2_c", "M_amp_c", "M_co2_e", "M_o2_c", "M_q8h2_c", "M_q8_c", "M_nadp_c", "M_nadph_c", "M_h2o_e", "M_nh4_e", "M_o2_e", "M_nh4_c"}
    unnecessaryMolecules = {"M_adp_c", "M_h2o_c", "M_co2_c", "M_h_c", "M_coa_c", "M_nad_c", "M_nadh_c", "M_h_e", "M_pi_e", "M_pi_c", "M_amp_c", "M_co2_e", "M_o2_c", "M_q8h2_c", "M_q8_c", "M_nadp_c", "M_nadph_c", "M_h2o_e", "M_nh4_e", "M_o2_e", "M_nh4_c"}
else:
    smallMolecules=set()
    exclude = {}

    # abundantMolecules = set()
    # unnecessaryMolecules = set()
    # # #exclude = {"M_atp_", "M_gtp_", "M_ctp_", "M_udp_", "M_ttp_", "M_utp_"}

    with open("./smallMolecules.txt", "r") as file:
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

    # 3.2 Get inhibitors if necesary
    
    unnecessaryMolecules, abundantMolecules = partitionNetworkHelper.getAbundantMolecules(smallMolecules, metabolicNetwork, exclude)

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

