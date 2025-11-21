# Packages 
import networkx as nx
import matplotlib.pyplot as plt
import sys
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import sympy as sp
from sympy import * 
import time 
from tqdm import tqdm
import concurrent.futures
from copy import copy
from copy import deepcopy


# 1. Functions for computation of distance matrices and ILP solivng


def computeExpectedShRed(ShReD:np.matrix):
    counter = 0
    P = np.zeros((np.shape(ShReD)[0], np.shape(ShReD)[1]))
    n = np.shape(P)[0]
    m = np.shape(P)[1]
    total = (n)**2 * (m)
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


def computeShortestPathMatrix(reactionNetwork:nx.DiGraph, noThreads:int):
    nodes = sorted(list((reactionNetwork.nodes())))
    n = len(nodes)
    A = np.zeros((n, n))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futureSet = {executor.submit(computShortestPath, reactionNetwork, nodes,  i) for i in range(n)}
        for future in tqdm(concurrent.futures.as_completed(futureSet), total=len(futureSet), leave = False, desc="ShortestPathMatrix"):
            try:
                resultsList, k = future.result()
                for l in range(len(resultsList)):
                    A[k][l] = resultsList[l]
            except Exception as exc:
                print('%r generated an exception: %s', exc)
    return A, nodes
#############################
#############################


def solveILP(G):
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gp.Model(env=env) as m:
            values = [-1,1]
            s = m.addVars(np.shape(G)[0], vtype=GRB.INTEGER, lb=-1, ub=1, name="s")
            Y = m.addVars(np.shape(G)[0], 2, vtype=GRB.BINARY, name = "Y")
            for i in range(np.shape(G)[0]):
                m.addConstr(gp.quicksum(Y[i,j] for j in range(2)) == 1)
                m.addConstr(s[i]==gp.quicksum(values[j] * Y[i,j] for j in range(2)))
            m.setObjective(gp.quicksum((G[i][j]*s[i]*s[j]) for j in range(np.shape(G)[0]) for i in range(np.shape(G)[1])), GRB.MAXIMIZE)
            #m.Params.PoolSearchMode = 2
            m.optimize()
            j=0
            v = np.zeros((1,np.shape(G)[0]))
            Q=m.getObjective().getValue()
            if m.SolCount > 1 and Q>0:
                k = 0
                while True:
                    m.setParam(GRB.Param.SolutionNumber, k)
                    for z in m.getVars():
                        if j>=np.shape(G)[0]:
                            break
                        v[0][j] = z.Xn
                        j+=1
                    oneVectorBool = True
                    for i in range(np.shape(v)[1]):
                        if v[0][i]!=1:
                            oneVectorBool = False
                            break
                    if oneVectorBool == False:
                        break
            else:
                for z in m.getVars():
                    if j>=np.shape(G)[0]:
                        break
                    v[0][j] = z.Xn
                    j+=1
                Q=m.getObjective().getValue()
    oneVectorBool = True
    for i in range(np.shape(v)[1]):
        if v[0][i] != 1:
            oneVectorBool = False
            break
    if oneVectorBool == True and Q>0:
        sys.exit("ERRRRORR!!!") 
    return v, Q
#############################
#############################


def correctG(G):
    for i in range(np.shape(G)[0]):
        for j in range(np.shape(G)[1]):
            if np.isnan(G[i][j]) or np.isinf(G[i][j]):
                G[i][j]=0
    return G
#############################
#############################


# 2. Matrix operations
def determinePositivAndNegativeSet(s, nodes:list):
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
def checkForCyclicity(posDG, negDG):
    cyclicity=True
    if nx.is_directed_acyclic_graph(posDG) == True or nx.is_directed_acyclic_graph(negDG)==True:
        cyclicity = False
    return cyclicity
#############################
#############################


def computeLeadingEigenvector(G:np.matrix):
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
    tNeg = True
    for i in range(np.shape(s)[1]):
        if round(largestEigenvector[i],5) <0:
            s[0][i] = -1
            t[0][i] = -1
        elif round(largestEigenvector[i],5) >0:
            s[0][i] = 1
            t[0][i] = 1
            sNeg = False
            tNeg = False
        else:
            tNeg = False
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


def computePartitioning(reactionNetwork:nx.DiGraph, noThreads:int):
    timeStamp = time.time()
    ShReD, nodes = computeShortestPathMatrix(reactionNetwork=reactionNetwork, noThreads=noThreads)
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
        s, Q, nodes = computePartitioning(reactionNetwork, noThreads)                                           # Perform computations for partitioning
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


