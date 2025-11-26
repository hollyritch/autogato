# Partitioning Helper 

# Packages 
import networkx as nx
import libsbml
import sys
import sympy as sp
from sympy import * 
import concurrent.futures
import numpy as np

def analyseOverlap(partitionTree:nx.DiGraph, network:nx.DiGraph, siblings:dict):
    ''' 
    Analyse the overlap between two generatted modules

    Upon invocation this functions determines the intersection between two sibling DiGraphs (modules) in the binary partition tree, in terms of metabolite vertices. The partition tree is thereby traversed in BFS, and a dictionary filled with a network as key and its sibling as value.

    Parameters
    ----------

    partiationTree : nx.DiGraph
        Is a a binary tree, that contains subgraphs of the reaction network of the metabolic network as nodes an directed edges from elder to child vertices, whenever the child is a proper subset of the elder

    network : nx.DiGraph 
        The metabolic network that was defined in the main function and represents the model to investigate.

    siblings : dictionary
        Specifying sibling relationsships for two children of the same elder in the partition tree.
    '''
    
    maxOverlap = 0
    overlapLengthDict = {}
    overlapMetaboliteDict = {}
    for subG in partitionTree.nodes():                                                  # Traverse the partitiontree
        subnetwork, metabolites, reactions = generateSubnetwork(subG, network)          # 
        if len(partitionTree.in_edges(subG))==0:                                        # Skip root vertex
            continue
        siblingSubG = siblings[subG]                                                    # read sibling of the current node
        if len(partitionTree.in_edges(subG))!=0:
            siblingSubnetwork, siblingMetabolites, siblingReactions = generateSubnetwork(siblingSubG, network)  # generate siblings netwwork
            overlapMetaboliteDict[(subG, siblingSubG)] = set(siblingMetabolites).intersection(set(metabolites))
            overlap = len(overlapMetaboliteDict[(subG, siblingSubG)])
            overlapLengthDict[overlap] = overlapLengthDict.setdefault(overlap, 0) + 1
            if overlap > maxOverlap:
                maxOverlap = overlap
    return maxOverlap, overlapLengthDict, overlapMetaboliteDict
#############################
#############################


def buildNetwork(model:libsbml.Model):
    '''Build the bipartite Digraph that represents the graph structure of the model
    
    Upon invocation, this function translates a libsbml (currently BIGG-models) into a networkX DiGraph object, by 
    translating each metabolite into a vertex, each reaction into a reaction vertex. Each vertex receives two label, specifying its name and the type - metabolite or reaction. A metabolite is connected to reaction by a directed edge whenever this metabolite is a reactant for a reaction and a reaction vertex is connected to a metablite by a directed edge whenever the metabolite is a product of this reaction yielding two types of edges, m -> r and r -> m edge, respectively. Corresponding stoichiometric coefficients are assigned as edge labels.
    
    Parameters
    ----------
    
    model : libsbml.Model
        Defines the metabolic model as a complex object, that allows to derive reactions, metabolites and all necessary information from it to build '''
    
    # 1. Define new variables
    metabolicNetwork = nx.DiGraph()
    
    # 2. Get list of reactions
    reactions = model.getListOfReactions()
    vertexIDs = {} 
    counter = 0
    for r in reactions:
        rName = r.getId()
        rFullName = r.getName()
        print(rName, rFullName)
        # if not rFullName.startswith("R_"):
        #     rFullName = "R_" + rFullName
        if "BIOMASS" in rName:                              # Exclude biomass function 
            continue
                                                            
        if "EX" in rName:                                   # Exlucde export and transport reaction since they do       
                                                            # not contribute to autocatalysis but only increase complexity
            continue
        if "transport" in rFullName.lower():
            continue
        reversible = r.getReversible()                              # Check for reversibility
        if reversible == True:                                      # If so split reactions in fw and reverse
            rNameFW = rName + "_fw"
        else:
            rNameFW = rName
        educts = r.getListOfReactants()                             # get reactants
        products = r.getListOfProducts()                            # get products
        reactionFWID = counter                                      # Assign id...which is just a counter
        counter+=1
        
        vertexIDs[rNameFW] = reactionFWID                           # Add reaction name and id to dict, so that they can 
                                                                    #be identified later on
        # Add reaction node to the network
        metabolicNetwork.add_node(reactionFWID)
        metabolicNetwork.nodes[reactionFWID]["Name"]=rNameFW
        metabolicNetwork.nodes[reactionFWID]["Type"]="Reaction"
        for e in educts:                                            # Add educts
            eSpecies = e.getSpecies()
            eductStoichiometry = e.getStoichiometry()
            # if not eSpecies.startswith("M_"):                       # this case should only appear in non-BIGG models
            #                                                         # done for unification of the models 
            #     eSpecies = "M_" + eSpecies
            if eSpecies not in vertexIDs:
                eSpeciesID = counter
                counter+=1
                vertexIDs[eSpecies] = eSpeciesID                    # add entry to id dict
                # Add educt node to networ
                metabolicNetwork.add_node(eSpeciesID)           
                metabolicNetwork.nodes[eSpeciesID]["Name"] = eSpecies
                metabolicNetwork.nodes[eSpeciesID]["Type"] = "Species"
            else:
                eSpeciesID = vertexIDs[eSpecies]
            
            # Add educt edge
            metabolicNetwork.add_edge(eSpeciesID, reactionFWID)
            metabolicNetwork.edges[eSpeciesID, reactionFWID]["Stoichiometry"]=eductStoichiometry
        for p in products:                                          # Add products
            pSpecies = p.getSpecies()
            productStoichiometry = p.getStoichiometry()
            # if not pSpecies.startswith("M_"):
            #     pSpecies = "M_" + pSpecies
            if pSpecies not in vertexIDs:
                pSpeciesID = counter
                counter+=1
                vertexIDs[pSpecies] = pSpeciesID                    # add product to ID dict
                
                # Add product to network
                metabolicNetwork.add_node(pSpeciesID)
                metabolicNetwork.nodes[pSpeciesID]["Name"] = pSpecies
                metabolicNetwork.nodes[pSpeciesID]["Type"] = "Species"
            else:
                pSpeciesID = vertexIDs[pSpecies]
            
            # Add product edge
            metabolicNetwork.add_edge(reactionFWID, pSpeciesID)
            metabolicNetwork.edges[reactionFWID, pSpeciesID]["Stoichiometry"]=productStoichiometry
        if reversible == True:                                      # Add the reaction node for the reverse edge and 
                                                                    # connect metabolites in reverse order to it as for the forwar reaction
            rNameRev = rName + "_rev"
            reactionRevID = counter                                 # asssign ID
            counter +=1
            vertexIDs[rNameRev] = reactionRevID                     # add reaction to ID dict

            # Add reaction node to network
            metabolicNetwork.add_node(reactionRevID)
            metabolicNetwork.nodes[reactionRevID]["Name"] = rNameRev
            metabolicNetwork.nodes[reactionRevID]["Type"] = "Reaction"
            for e in products:                                      # Now add edge edge (which are products for 
                                                                    # fw-reaction)
                eSpecies = e.getSpecies()
                s = e.getStoichiometry()
                eSpeciesID = vertexIDs[eSpecies]

                # Add educt edge
                metabolicNetwork.add_edge(eSpeciesID, reactionRevID)
                metabolicNetwork.edges[eSpeciesID, reactionRevID]["Stoichiometry"]=s
            for p in educts:                                        # Add products (educts in fw reaction)
                pSpecies = p.getSpecies()
                s = p.getStoichiometry()
                pSpeciesID = vertexIDs[pSpecies]

                # Add product edge
                metabolicNetwork.add_edge(reactionRevID, pSpeciesID)
                metabolicNetwork.edges[reactionRevID, pSpeciesID]["Stoichiometry"]=s
    return metabolicNetwork, vertexIDs                              # Return network and the id-dict which is structured 
                                                                    # vice versa
#############################
#############################


def createReactionNetwork(sCC:nx.DiGraph, reactions:set, inhibitors:dict):
    '''Create the R-Graph from the MR-Graph
    
        Upon invocation this function generates a DiGraph with only reaction vertices (no metabolite vertices). 
        Two reactions are connected by a directed edge if a product of the first is also an educt of the second. Inhibiting relationshipts, e.g. r1 inhibits r2 can also be included, but are not yet implemented.

    Parameters
    ----------

    sCC : nx.DiGraph
        Represents a strongly connected component of the input metabolic network after removal of unneccessary metabolites.

    reactions : set 
        Set of all reactions in the strongly connected component (sCC). 

    inhibitors : dict
        key : inhibiting reaction
        values : set of inhibited reactions
        A map specififying inhibing relationships, where the key reaction inhibits all reactions in the value.
    '''
    
    reactionNetwork = nx.DiGraph()
    reactions = sorted(list(reactions))
    for i in range(len(reactions)):
        r1 = reactions[i]
        for j in range(len(reactions)):
            if i==j:
                continue
            r2 = reactions[j]            
            # First Check if a product of r1 is a reactant of r2
            productsR1 = set(sCC.successors(r1))
            eductsR2 = set(sCC.predecessors(r2))
            if len(productsR1.intersection(eductsR2))>0:
                reactionNetwork.add_edge(r1, r2)
            # Now Check the converse
    for key, value in inhibitors.items():
        for in_edge in sCC.in_edges(key):
            r = in_edge[0]
            reactionNetwork.add_edge(r, value)
    return reactionNetwork
#############################
#############################


def getAbundantMolecules(smallMoleculesSet:set, metabolicNetwork:nx.DiGraph):
    unneccessaryMolecules =set()
    for n in metabolicNetwork.nodes():
        node = metabolicNetwork.nodes[n]["Name"]
        if node.startswith("M_"):
            if node.endswith("_e"):
                unneccessaryMolecules.add(node)
            shortendNode = "_".join(node.split("_")[:-1])+"_"
            if shortendNode in smallMoleculesSet:
                unneccessaryMolecules.add(node)
    return unneccessaryMolecules
#############################
#############################


def generateSubnetwork(subG:nx.Graph, metabolicNetwork:dict):
    ''' Generate a subnetwork of the MR network from a subnetwork of the reaction graph
    
    Upon invocation this function determines the MR (bipartite directed graph) from an undirected reaction network

    Parameters:
    ----------
    
    subG : undirected reaction network
        Specifies relationsships between reaction nodes in this subnetwork, i.e. (r,s)€ E <=> exits x: (r,x,s) is a path in the metabolic network

    metabolicNetwork : nx.DiGraph
        Specifies the model as a bipartite DiGraph.

    '''
    
    metabolites = set()                                                     # Create new set of overlapption metabolites
    subGReactions = set(subG.nodes())                                       # Retrive all nodes from the reaction network
    for r in subGReactions:                 
        for inEdge in metabolicNetwork.in_edges(r):                         # Get all educt metabolites
            metabolites.add(inEdge[0])
        for outEdge in metabolicNetwork.out_edges(r):                       # Get all prodict metabolites
            metabolites.add(outEdge[1])
    subnetwork = nx.subgraph(metabolicNetwork, metabolites.union(subGReactions)).copy() # generate subgraph from it.
    return subnetwork, sorted(list(metabolites)), sorted(list(subGReactions))
#############################
#############################


def getReactionsAndMetabolites(X:set, Y:set, metabolicNetwork:nx.DiGraph):
    for z in X:
        x = metabolicNetwork.nodes[z]["Name"]
        if x.startswith("R"):
            reactions = X
            metabolites = Y
        else:
            reactions = Y
            metabolites = X
        break
    return reactions, metabolites
#############################
#############################


def generateUndirectedReactionNetwork(reactionNetwork:nx.DiGraph):
    uRN = nx.Graph()
    for e in reactionNetwork.edges():
        uRN.add_edge(e[0],e[1])
    return uRN
#############################
#############################


def performCycleCongruenceCheck(newCs:list, cycleList:list):
    for i in range(len(newCs)):
        newC = newCs[i]
        for j in range(len(newC)):
            e = newC[j]
            if e.endswith("_in") or e.endswith("_out"):
                e = "_".join(e.split("_")[:-1])
                newC[j] = e
        newCs[i] = newC
    for i in range(len(cycleList)):
        c = cycleList[i]
        for j in range(len(c)):
            e = c[j]
            if e.endswith("_in") or e.endswith("_out"):
                e = "_".join(e.split("_")[:-1])
                c[j] = e
        cycleList[i] = c
    for i in range(len(cycleList)):
        c = cycleList[i]
        cSet = set(c)
        cSetBool = False
        for j in range(len(newCs)):
            newC = newCs[j]
            newCSet = set(newC)
            if newCSet == cSet:
                cSetBool = True
                break
        if cSetBool == False:
            print("Cycle that has no pendant is", cSet, c)
#############################
#############################


def performSanityChecks(leaves:set, partitionTree:nx.DiGraph, siblings:dict):
    lengthOfLeaves = 0
    leafSet1 = set()
    for G in leaves:
        lengthOfLeaves += len(G.nodes()) 
        for n in G.nodes():
            leafSet1.add(n)
    lengthOfLeaves2 = 0
    leafSet2 = set()
    for G in partitionTree.nodes():
        if len(partitionTree.out_edges(G)) == 0:
            lengthOfLeaves2 += len(G.nodes())
            for n in G.nodes():
                leafSet2.add(n)
    if lengthOfLeaves == lengthOfLeaves2:
        if len(leafSet1.symmetric_difference(leafSet2)) != 0:
            sys.exit("Automatically generated leaf set and leaves from the tree do not coincide although they have the same length")
    else:
        sys.exit("Automatically generated leaf set and leaves from tree do not have the same length")
    if not nx.is_tree(partitionTree):
        sys.exit("Partition tree is no tree!!!")
    for n in partitionTree.nodes():
        if len(partitionTree.out_edges(n))>0:
            if len(partitionTree.out_edges(n))!=2:
                sys.exit("Partition tree is no binary tree")
    for n, s in siblings.items():
        if len(partitionTree.in_edges(n))>1:
            sys.exit("Wrrrrroooooong, no tree!!!")
        for inEdge in partitionTree.in_edges(n):
            p = inEdge[0]
            for outEdge in partitionTree.out_edges(p):
                if n == outEdge[1]:
                    continue
                else:
                    putativeSibling = outEdge[1]
                    if s!=putativeSibling:
                        sys.exit("Siblings dictionary seems to be wrong")
#############################
#############################

