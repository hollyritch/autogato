# Packages 
import networkx as nx
import numpy as np
import time 
from tqdm import tqdm
import concurrent.futures
import sys


def computeExpectedShRed(ShReD:np.matrix):
    '''
    Translate ShRed to expected ShReD matrix

    Upon invocation this function translates the previously computed ShReD matrix into an expected ShReD matrix 
    as specified in "Identification of Biochemical Network Modules Based on Shortest Retroactive Distances, Sridharan et al., 2011, PLoS Computational Biology, 10.1371/journal.pcbi.1002262". In particular, an entry P[i,j] is the "arithmetic mean of the average of all non-zero and non-infinite ShReDs involving i and the average of all non-zero and non-infinite ShReDs involving j" (Sridharan et al.). For more details we refer to the original publication."

    Parameters
    ----------
    
    :param ShReD: numpy matrix with a distance measure between two reactions as specified in upper paper 
    :type ShReD: np.matrix

    Returns:
    - np.matrix: P - expected ShReD matrix
    '''
    P = np.zeros((np.shape(ShReD)[0], np.shape(ShReD)[1]))
    n = np.shape(P)[0]
    m = np.shape(P)[1]
    
    # Vorverarbeitung
    Di = np.zeros((n,1))
    Dj = np.zeros((m,1))
    
    for i in tqdm(range(n), leave = False, desc="expectedShRed PreComputation 1/3"):
        sum_i = 0
        D_i=0
        for k in range(m):
            if ShReD[i][k]!=0 and np.isinf(ShReD[i][k]) == False:
                D_i +=1
                sum_i += ShReD[i][k]
        if D_i != 0:
            Di[i] = sum_i/D_i

    for j in tqdm(range(m), leave = False, desc="expectedShRed PreComputation 2/3"):
        sum_j = 0
        D_j = 0
        for k in range(n):
            if ShReD[j][k]!=0 and np.isinf(ShReD[j][k]) == False:
                D_j +=1
                sum_j += ShReD[j][k]
        if D_j!=0:
            Dj[j] = sum_j/D_j
            
    for i in tqdm(range(n), leave = False, desc="expectedShRed PreComputation 3/3"):
        for j in range(m):
            if ShReD[i][j]==0:
                continue
            if np.isinf(ShReD[i][j])==True:
                P[i][j]=0
                continue
            P[i][j] = 1/2 * (Di[i] + Dj[j])
    return P           
#############################
#############################


def computShortestPath(reactionNetwork:nx.DiGraph, nodeList:list, i:int):
    '''
    Compute the shortest directed paths for a vertex to all other vertices in the input reactionNetwork and save them in a list.
    
    Parameters
    ----------

    :param reactionNetwork: R-Graph of the input metabolic network
    :type reactionNetwork: nx.DiGraph

    :param nodeList: list of all nodes in 
    :type nodeList: list of all reaction vertices in reactionNetwork

    :param i: index of the currently to be processed vertex in the node-list
    :type i: int

    Returns:
    - list : resultsList - list of distances between i and each vertex j
    - i : current index
    '''

    resultList = []
    n1 = nodeList[i]
    for j in range(len(nodeList)):
        if i == j: 
            continue
        else:
            n2 = nodeList[j]
            if nx.has_path(reactionNetwork,n1,n2)==True and nx.has_path(reactionNetwork, n2,n1):
                d = len(nx.shortest_path(reactionNetwork,n1,n2))+len(nx.shortest_path(reactionNetwork,n2,n1))-2
                resultList.append(d) 
            else:
                resultList.append(np.inf)
    return resultList, i
#############################
#############################


def computeShortestPathMatrix(reactionNetwork:nx.DiGraph):
    '''
    Generate a matrix which denotes the lenght of the shortest path between two vertices i, j in reactionNetwork

    Parameters
    ----------
    
    :param reactionNetwork: R-Graph of the input reaction Network
    :type reactionNetwork: nx.DiGraph

    Returns:
    - np.Matrix : A - shortest path matrix
    - list : nodes - sorted list of metabolites
    '''

    nodes = sorted(list((reactionNetwork.nodes())))
    n = len(nodes)
    A = np.zeros((n, n))
    if sys.platform.startswith("linux"):
        executor = concurrent.futures.ProcessPoolExecutor()
    elif sys.platform == "darwin":
        executor = concurrent.futures.ThreadPoolExecutor()
    else:
        executor = concurrent.futures.ProcessPoolExecutor()
    futureSet = {executor.submit(computShortestPath, reactionNetwork, nodes,  i) for i in range(n)}
    for future in tqdm(concurrent.futures.as_completed(futureSet), total=len(futureSet), leave = False, desc="ShortestPathMatrix"):
        try:
            resultsList, k = future.result()
            for l in range(len(resultsList)):
                A[k][l] = resultsList[l]
        except Exception as exc:
            print('%r generated an exception: %s', exc)
    executor.shutdown()
    return A, nodes
#############################
#############################


def correctG(G:np.matrix):
    '''
    Change NAN and infinity entries in the matrix G to 0.
    
    Parameters
    ----------

    :param G: G = P - ShReD (so called ShReD-based modularity matrix, as described in "Identification of Biochemical Network Modules Based on Shortest Retroactive Distances, Sridharan et al., 2011, PLoS Computational Biology, 10.1371/journal.pcbi.1002262".)
    :type G: np.matrix

    Return:
    - np.Matrix - corrected G
    '''

    for i in range(np.shape(G)[0]):
        for j in range(np.shape(G)[1]):
            if np.isnan(G[i][j]) or np.isinf(G[i][j]):
                G[i][j]=0
    return G
#############################
#############################


# 2. Matrix operations
def determinePositivAndNegativeSet(s:np.matrix, nodes:list):
    '''
    Determine the set of of positive and non-positive entries in the given vector s
    
    Paramters
    ---------

    :param s: leading (largest eigenvector) of the corrected matrix G computed elsewhere
    :type nodes: np.matrix

    :param nodes: List of nodes from the R-Graph
    :type nodes: list
    
    Returns:
    - set : positive set - the set of indices that refer to positive entries in s
    - set : negative set - the set of indices that refer to negative entries in s
    '''

    posSet, negSet = set(), set()
    for j in range(np.shape(s)[1]):
        if np.real(s[0][j])>0:
            posSet.add(nodes[j])
            #s[0][j]=1
        else:
            negSet.add(nodes[j])
            #s[0][j]=-1
    return posSet, negSet
#############################
#############################


# 3. Functions for network partitioning
def checkForCyclicity(posDG:nx.DiGraph, negDG:nx.DiGraph):
    '''
    Check if both in put graphs are acyclic

    Parameters
    ----------
    
    :param posDG: nx.DiGraph refering to the R-subgraph of the set of vertices with positive entries in the computed leading eigenvector 
    :type posDG: nx.DiGraph

    :param negDG: nx.DiGraph refering to the R-subgraph of the set of vertices with positive entries in the computed leading eigenvector 
    :type negDG: nx.DiGraph

    Returns:
    - bool : cyclicity - False if both are acyclic, true otherwise
    '''
    cyclicity=True
    if nx.is_directed_acyclic_graph(posDG) == True or nx.is_directed_acyclic_graph(negDG)==True:
        cyclicity = False
    return cyclicity
#############################
#############################


def computeLeadingEigenvector(G:np.matrix):
    '''
    Compute leading eigenvector of the input matrix G
    
    Parameters
    ----------

    :param G: ShReD-based modularity matrix (as described in "Identification of Biochemical Network Modules Based on Shortest Retroactive Distances, Sridharan et al., 2011, PLoS Computational Biology, 10.1371/journal.pcbi.1002262")
    :type G: np.matrix

    Returns:
    - np.matrix : v - leading (largest) eigenvector 
    - Q : float - Q = sum_{i^n} sum_{j^n} G[i,j]*s[i]*s[j]

    '''
    eigenvalues, eigenvectors = np.linalg.eigh(G)
    transposeEigenvectors = np.transpose(eigenvectors)
    largestEigenvalue = round(eigenvalues[0], 5)
    largestEigenvector = transposeEigenvectors[0]
    
    for i in range(1, len(transposeEigenvectors)):        
        if round(eigenvalues[i],10) > largestEigenvalue:
            largestEigenvalue = round(eigenvalues[i], 5)
            largestEigenvector = transposeEigenvectors[i]
        # else:
        #     if round(abs(eigenvalues[i]),10) > largestEigenvalue:
        #         largestEigenvalue = round(eigenvalues[i], 10)
        #         largestEigenvector = tEigenvectors[i]
    s = np.zeros((1,len(largestEigenvector)))
    t = np.zeros((1,len(largestEigenvector)))
    sNeg = True
    for i in range(np.shape(s)[1]):
        if round(largestEigenvector[i],5) <0:
            s[0][i] = -1
            t[0][i] = -1
        elif round(largestEigenvector[i],5) >0:
            s[0][i] = 1
            t[0][i] = 1
            sNeg = False
        else:
            s[0][i] = -1
            t[0][i] = 1
    t = np.negative(t)
    
    Qs = sum(sum(G[i][j]*s[0][i]*s[0][j] for j in range(np.shape(G)[1])) for i in range(np.shape(G)[0]))
    Qt = sum(sum(G[i][j]*t[0][i]*t[0][j] for j in range(np.shape(G)[1])) for i in range(np.shape(G)[0]))    
    
    # Now evaluate which of the two vectors maximizes the optimization function
    if round(Qt,5)>round(Qs,5):
        Q = Qt
        v = t
    elif round(Qt,5)<round(Qs,5):
        Q = Qs
        v = s 
    else: 
        if sNeg == False:
            Q = Qs
            v = s
        else:
            Q = Qt
            v = t
    return v, Q
#############################
#############################


def computePartitioning(reactionNetwork:nx.DiGraph):
    '''
    Compute all relevant functions for the partitioning of the input R-Graph
    
    :param reactionNetwork: R-Graph 
    :type reactionNetwork: nx.DiGraph

    Returns:
    - s : np.matrix - leading eigenvector of modularity based ShReD-Matrix G
    - Q : float - Q = sum_{i^n} sum_{j^n} G[i,j]*s[i]*s[j]
    - nodes : list of nodes from reactionNetwork
    '''
    
    ShReD, nodes = computeShortestPathMatrix(reactionNetwork=reactionNetwork)
    P = computeExpectedShRed(ShReD)                                               # Compute ShReD score
    G = P - ShReD                                                                 # Compute modified ShReD
    G = correctG(G=G)                                                             # Correct infinity values
    s, Q = computeLeadingEigenvector(G)
    return s, Q, nodes
#############################
#############################


def partitionNetwork(uReactionNetwork:nx.Graph, reactionNetwork:nx.DiGraph, partitionTree, cutOff:int, siblings:dict, leaves:set, noThreads:int):
    # First: Check if the network is weakly connected
    # 1. If not, then start separating the weakly connected components step by step from each other
    #    by taking the first as separate and then all others
    if nx.number_weakly_connected_components(reactionNetwork)>1:
        #print("Number of weakly connected components is", nx.number_weakly_connected_components(reactionNetwork))
        newWCCs = sorted(list(nx.weakly_connected_components(reactionNetwork)))
        firstSet = newWCCs[0]                                                                                   # Generate first set
        secondSet = set()       
        for i in range(1,len(newWCCs)):                                                                         # Generate second set
            secondSet = secondSet.union(newWCCs[i])
        firstDG = reactionNetwork.subgraph(nodes=firstSet).copy()                                               # Generate subgraphs of respective sets
        secondDG = reactionNetwork.subgraph(nodes=secondSet).copy()                                             #               -""-
        firstUG = uReactionNetwork.subgraph(nodes=firstSet).copy()                                                           #               -""-
        secondUG = uReactionNetwork.subgraph(nodes=secondSet).copy()                                                         #               -""-
        cyclicity = checkForCyclicity(firstDG, secondDG)                                                        # Check cyclcicity
        if cyclicity == True:                                                                                   # If cyclicity is maintained in both subnetworks:
            partitionTree.add_edge(uReactionNetwork, firstUG)                                                                # 1. add to partition tree and  
            partitionTree.add_edge(uReactionNetwork, secondUG)
            siblings[firstUG] = secondUG                                                                        # 2. Add to siblings dict
            siblings[secondUG] = firstUG
            if len(firstDG)>cutOff:                                                                             # 3. Try to partition: Check if the first graph is further partitionable
                partitionNetwork(firstUG, firstDG, partitionTree, cutOff, siblings, leaves, noThreads)                     # If so ... do it
            else:
                leaves.add(firstUG)                                                                             # Otherwise: Add the first graph as a leaf
            if len(secondDG)>cutOff:                                                                            # Check if the second graph is further partitionable
                partitionNetwork(secondUG, secondDG, partitionTree, cutOff, siblings, leaves, noThreads)                   # If so....do it
            else:
                leaves.add(secondUG)                                                                            # Otherwise add second network as a leaf
        else:                                                                                                   # If no cycle can be found in one of the paratitionings, add as leaf 
            leaves.add(uReactionNetwork)                                                         
    # 2. Start partitioning 
    else:
        s, Q, nodes = computePartitioning(reactionNetwork)                                           # Perform computations for partitioning
        if Q<=0:                                                                                                    # Don't continue partitioning if Q is less equal zero
            leaves.add(uReactionNetwork)
        else:                                                                                                       # Continue partitioning
            continuePartitioning(s, nodes, uReactionNetwork, partitionTree, cutOff, siblings, leaves, reactionNetwork, noThreads)
    return
#############################
#############################


def continuePartitioning(s:np.matrix, nodes:set, undirectedReactionNetwork:nx.Graph, partitionTree:nx.DiGraph, cutOff:int, siblings:dict, leaves:set, reactionNetwork:nx.DiGraph, noThreads:int):
    posSet, negSet = determinePositivAndNegativeSet(s, nodes)                                                   # Determine positive and negative sets according to ILP-solution 
    posUG = nx.subgraph(undirectedReactionNetwork, posSet).copy()                                               # Determine respective subgraophs
    posDG = nx.subgraph(reactionNetwork, posSet).copy()                                                         #               -""-
    negUG = nx.subgraph(undirectedReactionNetwork, negSet).copy()                                               #               -""-
    negDG = nx.subgraph(reactionNetwork, negSet).copy()                                                         #               -""-
    cyclicity = checkForCyclicity(posDG, negDG)                                                                 # Check if cycles are maintained in both partitionings
    if cyclicity == True:                                                                                       # If True:
        partitionTree.add_edge(undirectedReactionNetwork, posUG)                                                    # 1. Add undirected graphs of the RN to partition tree
        partitionTree.add_edge(undirectedReactionNetwork, negUG)                                                    #                       -""-
        siblings[posUG] = negUG                                                                                     # 2. Add to siblings dict
        siblings[negUG] = posUG                                                                                     #          -""-
        if len(posDG)>cutOff:                                                                                       # 3. If length of positive set is bigger than cutoff: 
            partitionNetwork(posUG, posDG, partitionTree, cutOff, siblings, leaves, noThreads)                                         # Continue partitioning
        else:
            #print("Leaves are smaller than cutoff, so we add", posUG)                                           # Else:       
            leaves.add(posUG)                                                                                               # Add undirected graph of positive set to leaves
        if len(negDG)>=cutOff:                                                                                      # 4. If length of positive set is bigger than cutoff: 
            partitionNetwork(negUG, negDG, partitionTree, cutOff, siblings, leaves, noThreads)                                         # Continue partitioning
        else:                                                                                                           # Else:
            #print("Leaves are smaller than cutoff, so we add", negUG)
            leaves.add(negUG)                                                                                               # Add undirected r-graph of negative set to leaves                   
    else:                                                                                                       # Otherwise: 
        #print("Cyclicity is not fullfilled anymore, so stop", undirectedReactionNetwork)
        leaves.add(undirectedReactionNetwork)                                                                               # Add undirected parent graph to leavs
    return
#############################
#############################


