# Partitioning Helper 

# Packages 
import networkx as nx
import libsbml
import sys
import sympy as sp
from sympy import * 
import concurrent.futures
import numpy as np


def buildNetwork(model:libsbml.Model):
    '''
    Build the bipartite Digraph that represents the graph structure of the model
    
    Upon invocation, this function translates a libsbml (currently BIGG-models) into a networkX DiGraph object, by 
    translating each metabolite into a vertex, each reaction into a reaction vertex. Each vertex receives two label, specifying its name and the type - metabolite or reaction. A metabolite is connected to reaction by a directed edge whenever this metabolite is a reactant for a reaction and a reaction vertex is connected to a metablite by a directed edge whenever the metabolite is a product of this reaction yielding two types of edges, m -> r and r -> m edge, respectively. Corresponding stoichiometric coefficients are assigned as edge labels.
    
    Parameters
    ----------

    :param model: libsbml.Model
        Defines the metabolic model as a complex object, that allows to derive reactions, metabolites and all necessary information from it to build 
    :type model: libsbml.Model
            
    Returns:
    - nx.DiGraph : metabolicNetwork - the directed bipartite graph of the chemical reaction network that was specified by the input sbml.
    - dict : vertexIDs - key: species idendtifier (BIGG), value: vertex identifier in metabolic network         
    '''
    
    # 1. Define new variables
    metabolicNetwork = nx.DiGraph()
    
    # 2. Get list of reactions
    reactions = model.getListOfReactions()
    vertexIDs = {} 
    counter = 0
    for r in reactions:
        rName = r.getId()
        rFullName = r.getName()
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
            if eSpecies not in vertexIDs:
                eSpeciesID = counter
                counter+=1
                vertexIDs[eSpecies] = eSpeciesID                    # add entry to id dict
                # Add educt node to network
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
            vertexIDs[rNameRev] = reactionRevID                     # Add reaction to ID dict

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
    '''
    Create the R-Graph from the MR-Graph
    
        Upon invocation this function generates a DiGraph with only reaction vertices (no metabolite vertices). 
        Two reactions are connected by a directed edge if a product of the first is also an educt of the second. Inhibiting relationshipts, e.g. r1 inhibits r2 can also be included, but are not yet implemented.

    Parameters
    ----------

    :param sCC: Represents a strongly connected component of the input metabolic network after removal of unneccessary metabolites.
    :type sCC: nx.DiGraph

    :param reactions: Set of all reactions in the strongly connected component (sCC). 
    :type reactions: set

    :param inhibitors:
        key : inhibiting reaction
        values : set of inhibited reactions
        A map specififying inhibing relationships, where the key reaction inhibits all reactions in the value.
    :type inhibitors: dict

        
    Returns:
    - nx.DiGraph : reactionNetwork - R-Graph of the input bipartite network
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
    '''
    Write a set of unnecessary molecules
    
    Upon invocation this function reads the set of small molecules and extracts all vertex identifiers within 
    the metabolic network that contain this identifier. Additionally, we also exclude all extracellular species
    to focus on intracellular metabolism and avoid unneccessary complexity.
    
    :param smallMoleculesSet: set of small molecules specified by the user that should be removed from the network.
    :type smallMoleculesSet: set
    :param metabolicNetwork: Directed bipartite graph that was given as an input sbml by the user earlier.
    :type metabolicNetwork: nx.DiGraph

    Returns:
    - set : unneccessary molecules - set of molecules that are going to be removed from the network.
    '''
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
    
    :param subG: 
        Specifies relationsships between reaction nodes in this subnetwork, i.e. (r,s)€ E <=> exits x: (r,x,s) is a path in the metabolic network
    :type subG: nx.Graph

    :param metabolicNetwork:
        Specifies the model as a bipartite DiGraph.
    :type metabolicNetwork: dict

    Returns:
    - subnetwork : nx.DiGraph - the directed bipartite graph induced by the set of reactions from the R-Graph subG, i.e. including all incoming and outcoming edges of each reaction vertex in subG and thereby also all metabolite vertices that are either reactant or metabolite of a reaction in subG

    - list : list of metabolites as vertex IDs sorted by their ID

    - list : list of reactions as vertex IDs sorted by their ID
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
    '''
    Identify reactions and metabolites

    Upon invocation this function identifies which of the disjoint vertex sets X and Y of a strongly connected components of the bipartite directed graph "metabolicNetwork" corresponds to set of metabolites and which to reactions.
    
    :param X: Set of vertices diksoint from X
    :type X: set
    :param Y: Set of vertices disjoint from X
    :type Y: set
    :param metabolicNetwork: Directed bipartite graph generated from the input sbml-file specified by the user earlier.
    :type metabolicNetwork: nx.DiGraph

    Returns:
    - set : reactions
    - set : metabolites
    '''
    for z in X:
        x = metabolicNetwork.nodes[z]["Type"]
        if x == "Reaction":
            reactions = X
            metabolites = Y
        else:
            reactions = Y
            metabolites = X
        break
    return reactions, metabolites
#############################
#############################

