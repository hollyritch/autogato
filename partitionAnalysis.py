import sys, os
import libsbml
import sympy as sp
import scipy as sc
import networkx as nx
from copy import deepcopy
from tqdm import tqdm
from copy import copy
import numpy as np
import concurrent.futures
import sys, os
import libsbml
import time
import pickle
import gc
from itertools import cycle
from itertools import chain
from collections import deque
from checkMatch import assembleCython, checkMatch, assembleCythonCores
import traceback


def analyzeCycles(G:nx.DiGraph, analyzeDict:dict, overlapDict:dict, childrensDict:dict, leaves:set, subN:nx.DiGraph, bound:int, species:str, treeCounter:int):
    # Global variables
    # 0.1. Get global variables
    global globalSubN 
    global cycleDict
    global E
    # 0.2 assign values
    globalSubN = subN
    circuitCounter = 0
    newGraphIdentifier = "G_" + str(len(cycleDict.keys()))
    cycleDict[newGraphIdentifier] = G
    with open("/scratch/richard/Autocatalysis/cycleData/"+ species +"/allCycles"+ str(treeCounter) +".txt", "a") as cycleFile:
        cycleFile.write(">>>" + newGraphIdentifier + "\n")
    if analyzeDict[G] == True:                                                                          # So, if you have to analyze the network
        if G in leaves:                                                                                 # Determine if it is a leaf
            if len(elemE)>0:
                print(len(elemE))
            if len(elemE)>1e6:
                print(len(elemE))
                sys.exit("Size of elementary circuits getting too large, please reduce the size of the network.")
            leafSimpleCycles = nx.simple_cycles(subN, length_bound = bound)
            circuitCounter = processCircuits(circuits = leafSimpleCycles, leaf = True, left = False, circuitCounter = circuitCounter)
        else:                                                                                           # Otherwise, we need to make sure to now separate the cycles
            leftChild = childrensDict[G]["left"]                                                        # Get left child
            rightChild = childrensDict[G]["right"]                                                      # Get right child
            overlapSet = overlapDict[G]                                                                 # Read overlap set
            overlapList = list(overlapSet)
            nodeDeleteList = []
            leaves = False
            for j in tqdm(range(len(overlapList)), leave=False, desc="Computing cycles for overlapping metabolites"):
                m = overlapList[j]    
                leftOutNetwork = getOutNetwork(outNetwork = leftChild, inNetwork = rightChild, startingNode = m, nodeDeletelist = nodeDeleteList)
                rightOutNetwork = getOutNetwork(outNetwork = rightChild, inNetwork = leftChild, startingNode = m, nodeDeletelist = nodeDeleteList)    
                nodeDeleteList.append(m)
                leftOutCircuits = nx.algorithms.cycles._bounded_cycle_search(leftOutNetwork, [m], length_bound=bound)
                rightOutCircuits = nx.algorithms.cycles._bounded_cycle_search(rightOutNetwork, [m], length_bound=bound)
                circuitCounter = processCircuits(circuits = leftOutCircuits, leaf = False, left = True, circuitCounter = circuitCounter)
                circuitCounter = processCircuits(circuits = rightOutCircuits, leaf = False, left = False, circuitCounter = circuitCounter) 
                if len(E)>0:
                    print(len(E))
                if len(elemE)>5e6*(12/noThreads):
                    print(len(E))
                    sys.exit("Size of elementary circuits getting too large, please reduce the size of the network.")
    return circuitCounter
#############################
#############################     


def analyzeElementaryCircuits(c:list):
    # Compute stochastic matrix and check metzler a
    mrEdgeSet = getEquivalenceClass(c)
    remove = determineDuplicates(c)
    return remove, c, mrEdgeSet
#############################
#############################


def analysePartitionTree(parameters, partitionTree:nx.DiGraph, siblings:dict, leaves:set, uRN:nx.Graph, network:nx.DiGraph, bound:int, species:str, treeCounter:int):
    global E
    global Q
    global elemE
    global cycleLengthDict
    global elementaryCircuits
    global equivClassLengthDict
    global checkNonMetzler
    global fluffleBool
    global coreBool
    global speedCores
    # 1. Define variables
    # 1.1 Dictionaries
    analyzeDict = {}                                                                # Dictionaries for storing information if an analysis on this node is necessary
    overlapDict = {}                                                                # Dictionary to store overlap of metabolites between to siblings                                                              # Dictionary to store all the cycles for subnetworks to impede enumerating them de nove
    childrensDict = {}
    cycleCounter = 0
    defineNewVariablesForParametersDictionary(parameters)
    # Define the three global variables
    # 2. Initialize datastructures for traversal of partition Tree
    for l in leaves: 
        analyzeDict[l] = True                       # Each leaf has to be analysed
        overlapDict[l] = set()                      # Each overlap of metabolites for laeves is 0
    current = list(copy(leaves))                    # Start with leaves
    visited = copy(leaves)                          # Store what has already been visited
    # 3. Define necessary variables
    while True:
        subG  = current.pop(0)                      # Get first element from the queue 
        rootBool= False
        if subG == uRN:
            rootBool = True
        else:
            subGSibling = siblings[subG]                # Read sibling of this element
            if subGSibling not in visited:              # If the sibling is not visited yet, then we cannot deal with this problem now, 
                current.append(subG)                    # Move current element to the end of the queue (we'll deal with it later) and go to the next one
                continue
        if rootBool == True:
            print("Now checking the root vertex, might take some time")
        subnetwork, metabolites, reactions = generateSubnetwork(subG, network)                                  # Generate subnetwork
        # First analyze subG
        additionalCycles = analyzeCycles(subG, analyzeDict, overlapDict, childrensDict, leaves, subnetwork, bound, species, treeCounter)
        # 2. Non-Metzler
        cycleCounter+=additionalCycles
        if rootBool == False:                       # Root boolean tells us, when we have arrived at the root....then we don't have to look for siblings
            current.remove(subGSibling)             # Since we are dealing now with the current element and its sibling, remove the sibling from the queue, otherwise we would do the same thin twice
            siblingSubnetwork, siblingMetabolites, siblingReactions = generateSubnetwork(subGSibling, network)      # Generate siblings subnetwork
            metaboliteOverlap = set(metabolites).intersection(set(siblingMetabolites))                              # Compute overlap of metabolites between the siblings networks for the parent
            for inEdge in partitionTree.in_edges(subG):                                                             # Determine if this is the root or not
                p = inEdge[0]
                childrensDict[p] = {}
                childrensDict[p]["left"] = subnetwork
                childrensDict[p]["right"] = siblingSubnetwork
                overlapDict[p] = metaboliteOverlap
                if len(metaboliteOverlap)>1:
                    analyzeDict[p] = True
                else:
                    analyzeDict[p] = False
                current.append(p)
                visited.add(p)
            # Second analyze subG Sibling
            additionalCycles = analyzeCycles(subGSibling, analyzeDict, overlapDict, childrensDict, leaves, siblingSubnetwork, bound, species, treeCounter)
            # Write results into corresponding lists 
            # 1 Metzler
            # 1.1 Autocatalytic
            # 1.2 Non-autocatalytic
            cycleCounter+=additionalCycles
        if len(current) == 0:
            break
    
    if len(parameters["autocatalyticMetzlerUnstableCycles"])>0:
        print("Number of elementary circuits", cycleCounter)
        print("Number of equivalence classes for elementary circuits", len(E))
        print()
        print("################## 1. Metzler-matrices ##################")
        print("#### 1.1 Autocatalytic ####")
        print("Number of elementary circuits associated with a Hurwitz-unstable Metzler Matrix (=autocatalytic)", len(parameters["autocatalyticMetzlerUnstableCycles"]))
        print("Number of elementary circuits associated with a Hurwitz-unstable Metzler Matrix and zero eigenvalue", len(parameters["autocatalyticMetzlerZeroDeterminantUnstableCycles"]))
        print("Number of elementary autocatalytic circuits associated with a Metzler matrix with a zero eigenvalue, but not Hurwitz unstable", len(parameters["autocatalyticMetzlerZeroDeterminantNotUnstableCycles"]))
        print()
        print("#### 1.2 Non-Autocatalytic ####")
        print("Number of non-autocatalytic Metzler matrices", len(parameters["nonAutocatalyticMetzlerCycles"]))
        print()
        print()
        print("################## 2. Non-Metzler-matrices ##################")
        print("#### 2.1 Autocatalytic ####")
        print("Number of autocatalytic elementary with a non-Metzler Matrix", len(parameters["notMetzlerAutocatalyticCycles"]))
        print("Number of autocatalytic elementary circuits with a non-Metzler Hurwitz-stable Matrix",len(parameters["notMetzlerStableAutocatalyticCycles"]))
        print("Number of autocatalytic elementary circuits with a non-Metzler Hurwitz-unstable Matrix",len(parameters["notMetzlerUnstableAutocatalyticCycles"]))
        print("Number of autocatalytic elementary circuits with a non-Metzler and a zero eigenvalue",len(parameters["notMetzlerNotStableNotUnstableAutocatalyticCycles"]))
        print()
        print("#### 2.2 Non-Autocatalytic ####")
        print("Number of non-autocatalytic non-Metzler Matrices",len(parameters["nonMetzlerNonAutocatalyticCycles"]))
        print()
        print()
    
    print("Start assembling larger cycles")
    elemE = deepcopy(E)
    parameters["elemCircuitEquivClasses"] = elemE
    parameters["elementaryCircuitsLengtDict"] = deepcopy(cycleLengthDict)
    parameters["equivClassElemCircuitsLength"] = deepcopy(equivClassLengthDict)
    parameters["elementaryCircuits"] = elementaryCircuits
    

#===========================================================================================
#                       Assemble fluffle equivalence classes
#===========================================================================================
    assemblyTimeStamp = time.time()
    print("Length of Q to check", len(Q))
    print("Length of E", len(E))
    if coreBool == False:
        assembleLargerEquivClassesParallel(parameters, Q, E, equivClassLengthDict)
    else:
        assembleCores(parameters, Q, E, speedCores)
    assemblyTime = time.time() - assemblyTimeStamp
    print("Time needed for equivalenceClasses", assemblyTime)

#===========================================================================================
#                       Check autocatalycity for fluffle equiv-classes
#===========================================================================================
    
    parameters["equivClassLength"] = equivClassLengthDict
    checkAutoCatTimeStamp = time.time()
    print("Length of equivClasses", len(E))
    if coreBool == False:
        cores = checkAutocatalycity(E)
        checkAutoCatTime=time.time()-checkAutoCatTimeStamp
        print("Time needed for checking autocatalycity", checkAutoCatTime)
    else:
        timeStampRealCores = time.time()
        realCores = checkUniquenessOfCores(speedCores)
        realCoresTime = time.time()-timeStampRealCores
        print("Number of cores", len(realCores), "time needed for inclusion", realCoresTime)

#===========================================================================================
#                           Enumerate fluffles 
#===========================================================================================
    if fluffleBool == True:
        print("Number of elementary circuits for fluffle enumeration")
        assemblyTimeStamp = time.time()
        assembleFluffles(parameters, elementaryCircuits)
        assemblyTime = time.time() - assemblyTimeStamp
        print("Time needed for fluffles", assemblyTime)
    
#===========================================================================================
#                           Write into parameters
#===========================================================================================

    parameters["allEquivClasses"] = E
    parameters["fluffleLengthsDict"] = cycleLengthDict
    parameters["fluffles"] = elementaryCircuits
    if coreBool==True:
        parameters["AutocatalyticCores"] = realCores
        print("Number of autocatalytic cores", len(realCores))
    else:
        parameters["AutocatalyticCores"] = cores
        for c in cores:
            for d in cores:
                if c<d:
                    print(c)
                    print(d)
                    sys.exit()
        print("Total number of equivalence classes:", len(E))
        print("Total number of fluffles:", sum(cycleLengthDict[l] for l in cycleLengthDict.keys()))
        #circuitEquivClasses, cNetEquivClasses, cycleIdDict = assembleLargerCores(parameters, queue=)
        # 2. Non-Autocatalytic
        parameters["cycleCounter"] = cycleCounter
        elemMetzlerCounter = 0
        metzlerUnstable = parameters["autocatalyticMetzlerUnstableCycles"]
        elemEquiv = parameters["elemCircuitEquivClasses"]
        
        for eq in metzlerUnstable:
            if eq in elemEquiv:
                elemMetzlerCounter+=1
        print("Number of total equivalence classes", len(E))
        #print("Number of all tested cycles", cycleCounter+additionalCycleCounter)
        print()
        print
        print("################## 1. Metzler-matrices ##################")
        print("#### 1.1 Autocatalytic ####")
        print("Number of autocatalytic cycles associated to an invertible Hurwitz unstable metzler matrix", len(parameters["autocatalyticMetzlerUnstableCycles"]), "amnog those", elemMetzlerCounter, "are repreented by elementary circuits.")
        print("Number of autocatalytic cycles associated to a Hurwitz unstable metzler matrix with zero determinant", len(parameters["autocatalyticMetzlerZeroDeterminantUnstableCycles"]))
        print("Number of autocatalytic cycles associated to a non-Hurwitz stable metzler matrix with zero determinant", len(parameters["autocatalyticMetzlerZeroDeterminantNotUnstableCycles"]))
        if coreBool==True:
            print("Number of autocatalytic cores", len(speedCores))
        else:
            print("Number of autocatalytic cores", len(parameters["AutocatalyticCores"]))

        print()
        print("#### 1.2 Non-autocatalytic ####")
        print("Number of non-autocatalytic Metzler Matrices", len(parameters["nonAutocatalyticMetzlerCycles"]))
        print()
        print()
        print("################## 2. Non-Metzler-matrices ##################")
        print("#### 2.1 Autocatalytic ####")
        print("Number of autocatalytic cycles associated to a non-Metzler Matrix", len(parameters["notMetzlerAutocatalyticCycles"]))
        print("Number of autocatalytic cycles associated to a Hurwitz-stable non-Metzler Matrix",len(parameters["notMetzlerStableAutocatalyticCycles"]))
        print("Number of autocatalytic cycles associated to a Hurwitz-stable non-Metzler Matrix",len(parameters["notMetzlerStableAutocatalyticCycles"]))
        print("Number of autocatalytic cycles associated to a Hurwitz-unstable non-Metzler Matrix",len(parameters["notMetzlerUnstableAutocatalyticCycles"]))
        print("Number of autocatalytic cycles associated to a non-unstable, non-stable non-Metzler Matrix (with a zero eigenvalue)", len(parameters["notMetzlerNotStableNotUnstableAutocatalyticCycles"]))
        print()
        print("#### 2.2 Non-Autocatalytic ####")
        print("Number of non-autocatalytic non-Metzler Matrices",len(parameters["nonMetzlerNonAutocatalyticCycles"]))
        print()
        print()
        print("Start categorizing types of autocatalysis")
        cycleCounter = 0
        for k,v in parameters["detAutocatalysis"].items():
            cycleCounter+=len(v)
        
        determineTypeOfAutocatalysis(parameters)
        print()
        print("Number of cycles in detAutocatalysis are", cycleCounter)
        print("Number of centralized autocatalytic cycles", len(parameters["centralized"]))
        print("Number of not centralized autocatalytic cycles", len(parameters["notCentralized"]))
#############################
#############################  


def callAssembleCython(equivClass:frozenset, equivClassValues:dict, cutoff:int):
    global elemE, M, circuitIdMrEdgeDict
    newEquivClasses, change = assembleCython(equivClass, equivClassValues,  elemE, M, circuitIdMrEdgeDict, cutoff)
    return newEquivClasses, change
#############################
#############################


def assemble(equivClass:frozenset, equivClassValues:dict, cutoff:int):
    global elemE
    intersecEquivClasses, changeE = getIntersectingEquivClassesParallel(equivClass)
    newEquivClasses = {}
    for interEqCl in intersecEquivClasses:
        newEquivClass = equivClass | interEqCl
        if len(newEquivClass)<=cutoff:
            newFrozen = frozenset(newEquivClass)
            interEquivClassValues = elemE[interEqCl]
            plausible, newMR = checkMatch(equivClassValues, interEquivClassValues)
            if plausible:
                newEquivClasses[newFrozen] = {"MR":newMR, "RM": 0, "Predecessors": {equivClass, interEqCl}, "Leaf": False, "Autocatalytic": False, "Metzler": True, "Update": False, "Visited": False, "Core": False}
    return newEquivClasses, changeE
#############################
############################# 


def assembleLargerEquivClassesParallel(parameters:dict, Q: deque, E:dict, equivClassLengthDict:dict):
    '''
    Determines all CS equivalence classes depending on the input set (here still Q)
    
    '''
    if len(Q)==0:                               
        return [], 0, []    
    # 0. Define new variables
    cutoff = parameters["cutoffLargerCycles"]
    noThreads = parameters["noThreads"]
    spinner = cycle("|/-\\")
    while True:
        if len(Q)>1e3:
            maxVal = min(int(2e12/len(Q)), len(Q))
            with concurrent.futures.ProcessPoolExecutor(max_workers=noThreads) as executor:
                for f in tqdm(concurrent.futures.as_completed(executor.submit(callAssembleCython, Q[i], E[Q[i]], cutoff) for i in range(maxVal)), total=maxVal, leave = False):
                    Q.popleft()
                    try:
                        newEquivClasses, change = f.result()
                        for c in change:
                            E[c]["Predecessors"].update(change[c]["Predecessors"])
                            E[c]["Leaf"] = False
                        for newFrozen in newEquivClasses:
                            values = newEquivClasses[newFrozen]
                            if newFrozen in E:
                                E[newFrozen]["Predecessors"].update(values["Predecessors"])
                            else:
                                E[newFrozen] = values
                                l = len(newFrozen)*2
                                equivClassLengthDict[l] = equivClassLengthDict.setdefault(l, 0)+1
                                if len(newFrozen)<cutoff:
                                    Q.append(newFrozen)
                        del f
                        del newEquivClasses
                        del change
                    except Exception as exc:
                        print('%r generated an exception: %s', exc)
                        print(traceback.format_exc())
        else:
            equivClass = Q.popleft()
            intersecEquivClasses = getIntersectingEquivClasses(equivClass, E)
            for interEqCl in intersecEquivClasses:
                newEquivClass = equivClass | interEqCl
                if len(newEquivClass)<=cutoff:
                    newFrozen = frozenset(newEquivClass)
                    if newFrozen in E:
                        E[newFrozen]["Predecessors"].update({equivClass, interEqCl})
                    else:
                        equivClassValues = E[equivClass]
                        interEqClValues = E[interEqCl]
                        plausible, newMR = checkMatch(equivClassValues, interEqClValues)
                        if plausible:
                            E[newFrozen] = {"MR":newMR, "RM": 0, "Predecessors": {equivClass, interEqCl}, "Leaf": False, "Autocatalytic": False, "Metzler": True, "Visited": True, "Core": False}
                            values = E[newFrozen]
                            l = len(newEquivClass)*2
                            equivClassLengthDict[l] = equivClassLengthDict.setdefault(l, 0)+1
                            if len(newEquivClass) < cutoff:
                                Q.append(newFrozen)
        sys.stdout.write(f"\r{next(spinner)} Queue length: {len(Q):<5}")
        sys.stdout.flush()
        if len(Q) == 0:
            break
    return
#############################
#############################  


def assembleFluffles(parameters:dict, queue:list):
    global checkNonMetzler
    global cycleLengthDict
    global elementaryCircuits
    if len(queue)==0:
        return [], 0, []
    # 0. Define new variables
    additionalCycleCounter = 0
    additionalCycles = []
    cutoff = parameters["cutoffLargerCycles"]*2
    # Generate important dictionaries for further analysis
    edgeCycleDict, cycleIDDict, cycleIDEdgeDict, cycleIDNodeDict, visitedEdges = generateEdgeCycleDict(queue)
    newQueue = deepcopy(list(cycleIDDict.keys()))
    initialNoCycles = len(cycleIDDict.keys())
    spinner = cycle("|/-\\")
    while True:
        cKey = newQueue.pop(0)
        c = cycleIDDict[cKey]
        intersectingCycles, edgesC = getIntersectingCycles(cKey, cycleIDEdgeDict, edgeCycleDict)
        for cIntID in intersectingCycles:
            cInt = cycleIDDict[cIntID]
            if cInt == c:                                                                           # if the cycle is identical to the one we are currently looking at
                continue
            else:
                edgesC2 = cycleIDEdgeDict[cIntID]                                                   # Read edges for the cycle we are looking at 
                allEdges = frozenset(edgesC.union(edgesC2))                                         # Generate set of all edges of c1 and c2
                additionalCycleCounter +=1                                                          # Increase number of analysed cycles
                if allEdges in visitedEdges:                                                        # If we have seen this set of edges already, then continue
                    continue
                if cKey >= initialNoCycles:
                    cNew = [*c,cInt]
                else:
                    cNew = [c,cInt]
                additionalCycles.append(cNew)
                visitedEdges.add(allEdges)
                plausible = checkPlausibilityOfMatching(parameters, cKey, cIntID, cycleIDEdgeDict, cycleIDNodeDict)
                if plausible == True:
                    if len(cNew)<=cutoff:
                        cNewSet = set()
                        for circ in cNew:
                            for v in circ:
                               cNewSet.add(v) 
                        newKey = len(cycleIDDict.keys())
                        newQueue.append(newKey)
                        elementaryCircuits.append(cNew)
                        cycleIDDict[newKey]= cNew
                        cycleIDEdgeDict[newKey] = cycleIDEdgeDict[cKey].union(cycleIDEdgeDict[cIntID])
                        cycleIDNodeDict[newKey] = cycleIDNodeDict[cKey].union(cycleIDNodeDict[cIntID])
                else:
                    continue
                    #print("Unfortnuately not plausible")
        sys.stdout.write(f"\r{next(spinner)} Queue length: {len(newQueue):<5}")
        sys.stdout.flush()
        if len(newQueue) == 0:
            break
    return additionalCycleCounter
#############################
#############################  


def checkAutocatalycityRecursive(E:dict, equivClass:frozenset, equivClassProperties:dict, cores:set):
    S, metzler = computeSubstochasticMatrixForSetOfMREdges(parameters, equivClass)
    if equivClassProperties["Leaf"] == True:                                                                   # It is a leaf                                                                   
        aM, nAM, autocatalytic = determineCircuitPropertiesMetzler(equivClass, S) 
        writeDataToListsAndDicts(parameters, aM, nAM, [], [])
        equivClassProperties["Autocatalytic"] = autocatalytic
        equivClassProperties["S"]=S
        equivClassProperties["Core"] = True
        if autocatalytic==True:
            cores.add(equivClass)
    else:
        autocatalytic = False
        predecessors = equivClassProperties["Predecessors"]
        for p in predecessors:
            pValue = E[p]
            if pValue["Visited"] == False:
                checkAutocatalycityRecursive(E, p, pValue, cores)
            if pValue["Autocatalytic"] == True:
                autocatalytic = True
                equivClassProperties["Autocatalytic"] = True
                writeDataToListsAndDicts(parameters, [equivClass], [], [], [])
                break
        if autocatalytic == False:
            aM, nAM, autocatalytic = determineCircuitPropertiesMetzler(equivClass, S) 
            if autocatalytic == True:
                equivClassProperties["Autocatalytic"] = True
                equivClassProperties["S"]=S
                equivClassProperties["Core"] = True
                cores.add(equivClass)
                writeDataToListsAndDicts(parameters, [equivClass], [], [], [])
            else:
                writeDataToListsAndDicts(parameters, [], [equivClass], [], [])
    equivClassProperties["Visited"] = True
    return
#############################
############################# 


def checkAutocatalycity(E:dict):
    '''Classifies all CS matrices collected so far, if they are autocatatalytic or not'''
    cores = set()
    global checkNonMetzler
    for equivClass in tqdm(E, leave = False, desc="Checking autocatalycity"):
        equivClassProperties = E[equivClass]  
        if equivClassProperties["Visited"] == True:
            continue
        S, metzler = computeSubstochasticMatrixForSetOfMREdges(parameters, equivClass)
        if equivClassProperties["Leaf"] == True:                                                                   # It is a leaf
            if metzler == True:                                                                     
                aM, nAM, autocatalytic = determineCircuitPropertiesMetzler(equivClass, S) 
                writeDataToListsAndDicts(parameters, aM, nAM, [], [])
                equivClassProperties["Autocatalytic"] = autocatalytic
                equivClassProperties["S"]=S
                if autocatalytic == True:
                    cores.add(equivClass)
            else:
                equivClassProperties["Autocatalytic"] = False
                equivClassProperties["Metzler"] = False
                equivClassProperties["S"] = S
                writeDataToListsAndDicts(parameters, [], [], [], [equivClass])
        else:
            if metzler==True:
                autocatalytic = False
                predecessors = equivClassProperties["Predecessors"]
                for p in predecessors:
                    pValue = E[p]
                    if pValue["Visited"] == False:
                        checkAutocatalycityRecursive(E, p, pValue, cores)
                    if pValue["Autocatalytic"] == True:
                        autocatalytic = True
                        equivClassProperties["Autocatalytic"] = True
                        equivClassProperties["S"] = S
                        writeDataToListsAndDicts(parameters, [equivClass], [], [], [])
                        break
                if autocatalytic == False:
                    aM, nAM, autocatalytic = determineCircuitPropertiesMetzler(equivClass, S) 
                    if autocatalytic == True:
                        equivClassProperties["Autocatalytic"] = True
                        equivClassProperties["Core"] = True
                        cores.add(equivClass)
                        writeDataToListsAndDicts(parameters, [equivClass], [], [], [])
                    else:
                        writeDataToListsAndDicts(parameters, [], [equivClass], [], [])
            else:
                if checkNonMetzler==True:
                    nMAC, nMnAC, autocatalytic = determineCircuitPropertiesNonMetzler(equivClass, S)
                    writeDataToListsAndDicts(parameters, [], [], nMAC, nMnAC)
        equivClassProperties["Visited"] = True
    return cores
#############################
#############################


def checkEquivalenceClass(c:list, eqClass:set):
    """
    Classify elementary circuits into equivalence classes according their 
    vertices and edges.

    On invocation, this function determines information about the vertices 
    and edges of elementary circuits given on the list queue. Importantly,
    elementary circuits vertex and edge-sets (E and E1: only metabolite -> reaction edges) serve 
    as keys and elementary circuits are stored according to equivalence classes
    invoked by these properteis in different dictionaries.
    
    Parameters
    ----------
    Q : list
        Contains all elementary circuits enumerated by the Johnsons-Algorithm in 
        def(analysePartitionTree).

    E : dict 
        Key: (frozen sets) of equivalence classes composed of MR-edges
        Value: tuple(metabolites, reactions)

    M : dict
        Key: One MR-edge
        Value: Set of elementary circuits containing this particular MR-edge

    cycleIDDict : dict
        Key: Integer (subsequently termed cycle-identifier)
        Value: Elementary circuit
    
    circuitIdMrEdgeDict : dict
        Key: cycle-identifier
        Value: Set of MR-edges

    """
    # 0. Read Variables
    global E                                                                        # \mathcal{E} im paper
    global Q 
    global M                                                                        # Dictionary Key: One! MR-edge only, Value: Set of cycles containing this MR-edge
    global circuitIdDict                                                            # Dictionary Key: cycle-identifier (int), Value: Cycle
    global circuitIdMrEdgeDict                                                      # Dictionary Key: Cycle-Identifier Value: Set of MR-edges contained in this cycle    
    # 1. Iterate over all cycles
    i = len(Q)
    # 2. Iterate only over MR-edges
    frozenEq = frozenset(eqClass)
    if frozenEq not in E:                                           # If MR-edgeset has not been added to bump-equivalence classes of elem. circuits, add it to new Q 
        mr = {}
        rm = {}
        Q.append(frozenEq)
        for e in eqClass:
            M.setdefault(e, set()).add(i)                           # Add edge if not already done to mrEdgeCycleDict with value of the current elementary circuit
            mr[e[0]]=e[1]
            rm[e[1]]=e[0]
        circuitIdMrEdgeDict[i] = frozenEq                           # Remember for this particular elementary circuit, which     
        E[frozenEq] = {"MR": mr, "RM": rm, "Predecessors": set(), "Leaf": True, "Autocatalytic": False, "Metzler": True, "Visited": False, "Core": False}     # Add new equivalence class to E with corresponding dictionary for M-R and R-M relationship, a set for precursors, and four flags: autocatalytic, Metzler, leaf, visited
        circuitIdDict[i] = c
        return True
    else:
        return False
#############################
#############################  


def checkPlausibilityOfMatching(parameters:dict, c1Key:int, c2Key:int, cycleIDEdgeDict:dict, cycleIDNodeDict:dict):
    metabolicNetwork = parameters["metabolicNetwork"]
    edgesC1 = cycleIDEdgeDict[c1Key]
    edgesC2 = cycleIDEdgeDict[c2Key]
    nodesC1 = cycleIDNodeDict[c1Key]
    nodesC2 = cycleIDNodeDict[c2Key]
    intersectingEdges = edgesC1.intersection(edgesC2)
    plausible = True
    G = nx.DiGraph()
    G.add_edges_from(intersectingEdges)
    if len(G)==0:
        return False
    intersectingNodes = nodesC1.intersection(nodesC2)
    for n in intersectingNodes:
        G.add_node(n)
    for c in nx.weakly_connected_components(G):
        intersectingGComponent = G.subgraph(c).copy()
        if len(intersectingGComponent.edges())%2==0:
            return False
        else:
            for n in intersectingGComponent.nodes():
                if len(intersectingGComponent.in_edges(n))==0:
                    if metabolicNetwork.nodes[n]["Type"] == "Reaction":
                    #if node.startswith("R_"):
                        return False
                if len(intersectingGComponent.out_edges(n))==0:
                    if metabolicNetwork.nodes[n]["Type"]=="Species":
                    #if n.startswith("M_"):
                        return False
    return plausible
#############################
#############################


def computeSubstochasticMatrixForSetOfMREdges(parameters:dict, newEquivClass:set):
    '''
        Determine CS matrix
    
        Upon invocation this function determines the k x k CS matrix for a set of MR-edges, 
        i. e. a submatrix of the stochastic matrix S, where columns are re-ordered according 
        to the given MR relationship. Accordingly, for an edge (m,r) then r represents the i-th
        column if and only if m represents the i-th row.

        Parameters
        ----------
        
        parameters : dict
            Key: Str
            Value: arbitrary data structures accumulated during the execution of the whole module

        newEquivClass : set
            Set of metabolite - reaction edges
        
        k : int
            Number of metabolites and reactions

        S : Sympy Matrix 
            This matrix represents the big stoichiometric matrix

        mID : dict
            Key : str (metabolite)
            Value : int (row corresponding to the metabolite in S)
        
        rID : dict
            Key : str (reaction)
            value : int (column corresponding to the reaction in S)
            
        metzler : boolean 
            Determines whether the CS matrix is Metzler or not
        
        subS : Numpy Matrix
            CS Matrix, to be filled, initiated with zeros.

        mRDict : dict
            Specifies the relationship in terms of rows and columns between the 
            metabolite and reaction of an MR edge 
    '''

    S = parameters["StoichiometricMatrix"]
    mID = parameters["mID"]
    rID = parameters["rID"]
    metzler = True
    k = len(newEquivClass)
    subS = np.zeros((k,k))
    mRDict = {}
    for e in newEquivClass:
        mRDict[mID[e[0]]] = rID[e[1]]
    sortedMetabolites = sorted(mRDict.keys())
    for i in range(k):
        mIDRow = sortedMetabolites[i]
        for j in range(k):
            mIDCol = sortedMetabolites[j]
            rIDCol = mRDict[mIDCol]
            subS[i][j] = S[mIDRow, rIDCol]
            if i != j and subS[i][j]<0:
                metzler = False
            if i==j:
                if subS[i][j]>=0:
                    sys.exit("ERRRRRRORR, CS matrix is not a CS matrix")
    return subS, metzler
#############################
#############################


def determineDuplicates(cycle:list):
    global parameters
    visited = set()
    metabolicNetwork = parameters["metabolicNetwork"]
    cycle0 = cycle[0]
    if metabolicNetwork.nodes[cycle0]["Type"] == "Species":
    #if cycle[0].startswith("M_"):
        offSet = 0
    else:
        offSet = 1
    for i in range(offSet, len(cycle)):
        e = cycle[i]
        if type(e)==str:
                e = int("_".join(e.split("_")[:-1]))
                cycle[i] = e
        if e in visited:
            return True
        visited.add(e)
    return False
#############################
#############################


def defineNewVariablesForParametersDictionary(parameters:dict): 
    parameters["detAutocatalysis"] = {}
    # 1.2 Lists
    # 1.2.1 Lists for Metzler cycles 
    parameters["autocatalyticMetzlerUnstableCycles"] = []
    parameters["autocatalyticMetzlerUnstableInvertibleCycles"] = []
    parameters["autocatalyticMetzlerZeroDeterminantCycles"] = []
    parameters["autocatalyticMetzlerZeroDeterminantUnstableCycles"] = []
    parameters["autocatalyticMetzlerZeroDeterminantNotUnstableCycles"] = []
    parameters["nonAutocatalyticMetzlerCycles"] = []
    # 1.3 Lists for non Metzler-Lists
    parameters["notMetzlerAutocatalyticCycles"] = []
    parameters["notMetzlerUnstableAutocatalyticCycles"] = []
    parameters["notMetzlerNotUnstableAutocatalyticCycles"] = []
    parameters["notMetzlerNotStableNotUnstableAutocatalyticCycles"] = []
    parameters["notMetzlerStableAutocatalyticCycles"] = []
    parameters["nonMetzlerNonAutocatalyticCycles"] = []
    return
#############################
############################# 


def determineAutocatalycityNonMetzler(S:np.matrix):
    k = sp.shape(S)[0]
    A = (-1)*S
    b = np.zeros((k,1))    
    c = np.ones((1,k))
    bounds = []
    for i in range(k):
        bounds.append((1, None)) 
        b[i][0]=-1
    result = sc.optimize.linprog(c=c, A_ub=A, b_ub=b, bounds=bounds)
    if result["success"]==True:
        return True
    else:
        return False
#############################
############################# 


def determineCircuitPropertiesMetzler(c:set, S:np.matrix):
    aM = []
    nAM = []
    unstable = determineStability(T=S)
    if unstable == True:
        aM.append(c)
    else:
        nAM.append(c)
    return aM, nAM, unstable
#############################
############################# 


def determineCircuitPropertiesNonMetzler(c:set,  S:np.matrix):
    # 0. Read variables from parameters dict
    nMAC = []
    nMnAC = []
    autocatalytic = determineAutocatalycityNonMetzler(S)
    if autocatalytic == True:
        nMAC.append(c)
    else:
        nMnAC.append(c)
    return nMAC, nMnAC, autocatalytic
#############################
############################# 


def determineStability(T:np.matrix):
    k = np.shape(T)[0]
    unstable = False
    if k>=3:
        sM = sc.sparse.csr_matrix(T)
        lamda = sc.sparse.linalg.eigs(sM, k=1, which = "LR", return_eigenvectors = False)
        if round(np.real(lamda[0]), 5)>0:
            unstable = True   
    else:
        for lamda in np.linalg.eigvals(T):
            if round(np.real(lamda),5)>0:        
                unstable = True
                break
    del T
    return unstable
#############################
#############################


def determineTypeOfAutocatalysis(parameters:dict):
    metzlerAutocatalytic = parameters["autocatalyticMetzlerUnstableCycles"]
    metabolicNetwork = parameters["metabolicNetwork"]
    E = parameters["allEquivClasses"]
    centralizedCircuits = []
    notCentralizedCircuits = []
    for eqC in metzlerAutocatalytic:
        mR = E[eqC]["MR"]
        metabolites = set(mR.keys())
        reactions = set(mR.values())
        vertices = set(metabolites.union(reactions))
        subgraph = nx.subgraph(metabolicNetwork, vertices)
        circuits = list(nx.simple_cycles(subgraph))        
        for m in metabolites:
            centralized=True
            for c in circuits:
                if m not in c:
                    centralized=False
                    break
            if centralized==True:
                break
        if centralized==True:
            centralizedCircuits.append(eqC)
        else:
            notCentralizedCircuits.append(eqC)
    parameters["centralized"] = centralizedCircuits
    parameters["notCentralized"] = notCentralizedCircuits
    return 
#############################
#############################  


def generateEdgeCycleDict(queue):
    edgeCycleDict = {}
    cycleIDDict = {}
    cycleIDEdgeDict = {}
    cycleIDNodeDict = {}
    visitedEdges = set()
    for i in range(len(queue)):
        c = queue[i]
        cycleIDDict[i] = c
        edgeSet = set()
        nodeSet = set()
        for j in range(len(c)):
            nodeSet.add(c[j])
            if j == len(c)-1:
                e = (c[j], c[0])
            else:
                e = (c[j], c[j+1])
            edgeSet.add(e)
            if e in edgeCycleDict.keys():
                edgeCycleDict[e].add(i)
            else:
                edgeCycleDict[e] = {i}
        cycleIDEdgeDict[i] = edgeSet
        cycleIDNodeDict[i] = nodeSet
        visitedEdges.add(frozenset(edgeSet))
    return edgeCycleDict, cycleIDDict, cycleIDEdgeDict, cycleIDNodeDict, visitedEdges
#############################
############################# 


def generateStoichiometricMatrix(parameters:dict, model:libsbml.Model):
    # 0. Read Parameters
    mID = parameters["mID"]
    rID = parameters["rID"]
    mList = parameters["mList"]
    rList = parameters["rList"]
    metabolicNetwork = parameters["metabolicNetwork"]
    # 1. Assign new variables
    n = len(mID)
    m = len(rID)
    S = np.zeros((n, m))
    # 2. Generate Stoichiometric Matrix
    for i in range(n):
        metaboliteID = mList[i]
        metabolite = metabolicNetwork.nodes[metaboliteID]["Name"]
        for j in range(m):
            reactionID = rList[j]
            r = metabolicNetwork.nodes[reactionID]["Name"]
            forward = True
            if r.split("_")[-1]=="rev": 
                r = "_".join(r.split("_")[0:-1])
                forward = False
            if r.split("_")[-1]=="fw": 
                r = "_".join(r.split("_")[0:-1])
            rObject = model.getReaction(r)
            if forward == True:
                educts = rObject.getListOfReactants()
                products = rObject.getListOfProducts()
            else:
                educts = rObject.getListOfProducts()
                products = rObject.getListOfReactants()
            '''TODO(Change this  to)'''
            for e in educts:
                if metabolite == e.getSpecies():
                    S[i][j] = - metabolicNetwork.edges[metaboliteID, reactionID]["Stoichiometry"]
            for p in products:
                if metabolite == p.getSpecies():
                    S[i][j] = metabolicNetwork.edges[reactionID, metaboliteID]["Stoichiometry"]
    return S
#############################
#############################


def generateSubnetwork(subG:dict, metabolicNetwork:dict):
    metabolites = set()
    subGReactions = set(subG.nodes()) 
    for r in subGReactions:
        for inEdge in metabolicNetwork.in_edges(r):
            metabolites.add(inEdge[0])
        for outEdge in metabolicNetwork.out_edges(r):
            metabolites.add(outEdge[1])
    subnetwork = nx.subgraph(metabolicNetwork, metabolites.union(subGReactions)).copy()
    return subnetwork, sorted(list(metabolites)), sorted(list(subGReactions))
#############################
#############################


def getIntersectingCycles(cKey:int, cycleIDEdgeDict:int, edgeCycleDict:dict):
    edgesC = cycleIDEdgeDict[cKey]
    intersectingCycles = set()
    for e in edgesC:
        intersectingCycles = intersectingCycles.union(edgeCycleDict[e])
    return intersectingCycles, edgesC
#############################
#############################


def getEquivalenceClass(c:list):
    """
    Classify elementary circuits into equivalence classes according their 
    vertices and edges.

    On invocation, this function determines information about the vertices 
    and edges of elementary circuits given on the list queue. Importantly,
    elementary circuits vertex and edge-sets (E and E1: only metabolite -> reaction edges) serve 
    as keys and elementary circuits are stored according to equivalence classes
    invoked by these properteis in different dictionaries.
    
    Parameters
    ----------
    c  : list 
        List of vertices (metabolite, reaction, metabolite, reaction, ...), i.e. an elementary circuit as a list of nodes

    """

    # 0. Define Variables
    global parameters
    mrEdgeSet = set()
    metabolicNetwork = parameters["metabolicNetwork"]  
    # 2. Iterate only over MR-edges
    offset=0
    if type(c[0])!=str:
        if metabolicNetwork.nodes[c[0]]["Type"] == "Reaction":
    # if c[0].startswith("R_"):
            offset = 1
    n = int(len(c)/2)        
    for j in range(n):
        if j==n-1 and offset==1:
            e = (c[j*2+offset], c[0])    
        else:
            e = (c[j*2+offset], c[j*2+offset+1])
        m = e[0]
        if type(m) == str:
            m = int("_".join(m.split("_")[:-1]))
        else:
            if metabolicNetwork.nodes[m]["Type"]=="Reaction":
                sys.exit("Somethings wrong here with the edge detection") 
        mrEdgeSet.add((m, e[1]))                                                        # Add edge to growing MR-edge set
    return mrEdgeSet
#############################
#############################  


def getIDDicts(metabolites:set, reactions:set):
    mID, rID, iDM, iDR  = {}, {}, {}, {}
    mList, rList = [], []
    i=0
    for m in metabolites:
        mID[m] = i
        iDM[i] = m
        mList.append(m)
        i+=1
    j = 0
    for r in reactions:
        rID[r] = j
        iDR[j] = r
        rList.append(r)
        j+=1
    return mID, rID, iDM, iDR, mList, rList
#############################
#############################


def getIntersectingEquivClassesParallel(equivClass:set):
    '''Determine the those equivalence classes (sets of MR-edges) that intersect with the current cycle of interest.
    
    Parameters
    ----------

    equivClass : set
        Set of MR-edges of the current cycle to check 
    
        circuitIdMrEdgeDict : dictionary
            Key: cycle identifier (e.g. ckey)
            Value: set of metabolite-reaction edges 

        M : dictionary
            Key: frozenset of metabolite-reaction (MR) edges
            Value: Set of cycles corresponding exhibiting these MR-edges
    '''
    global M, circuitIdMrEdgeDict
    intersecEquivClasses = []                                                           # Initiate empty set to 
    changeE = {}
    all_circuits = set(chain.from_iterable(M[e] for e in equivClass))
    for c in all_circuits:            
        cEqCl = circuitIdMrEdgeDict[c]
        if equivClass == cEqCl:
            continue
        elif equivClass < cEqCl:
            value = changeE.setdefault(cEqCl, {})
            value.setdefault("Predecessors", set()).add(equivClass)
            value["Leaf"] = False
        elif cEqCl < equivClass:
            value = changeE.setdefault(equivClass, {})
            value.setdefault("Predecessors", set()).add(cEqCl)
            value["Leaf"] = False
        else:
            intersecEquivClasses.append(cEqCl)    
    return intersecEquivClasses, changeE
#############################
#############################


def getIntersectingEquivClasses(equivClass:set, E:dict):
    '''Determine the those equivalence classes (sets of MR-edges) that intersect with the current cycle of interest.
    
    Parameters
    ----------

    equivClass : set
        Set of MR-edges of the current cycle to check 
    
        circuitIdMrEdgeDict : dictionary
            Key: cycle identifier (e.g. ckey)
            Value: set of metabolite-reaction edges 

        M : dictionary
            Key: frozenset of metabolite-reaction (MR) edges
            Value: Set of cycles corresponding exhibiting these MR-edges
    '''
    global M 
    global circuitIdMrEdgeDict
    intersecEquivClasses = []                                                           # Initiate empty set to 
    all_circuits = set(chain.from_iterable(M[e] for e in equivClass))
    for c in all_circuits:            
        cEqCl = circuitIdMrEdgeDict[c]
        if cEqCl in E[equivClass]["Predecessors"]:
            continue
        elif equivClass in E[cEqCl]["Predecessors"]:
            continue
        elif equivClass == cEqCl:
            continue
        elif equivClass < cEqCl:
            value = E.setdefault(cEqCl, {})
            value.setdefault("Predecessors", set()).add(equivClass)
            value["Leaf"] = False
        elif cEqCl < equivClass:
            value = E.setdefault(equivClass, {})
            value.setdefault("Predecessors", set()).add(cEqCl)
            value["Leaf"] = False
        else:
            intersecEquivClasses.append(cEqCl)    
    return intersecEquivClasses
#############################
#############################


def getOutNetwork(outNetwork:nx.DiGraph, inNetwork:nx.DiGraph, startingNode:str, nodeDeletelist:list):
    global globalSubN 
    newNetwork = deepcopy(globalSubN )
    for n in nodeDeletelist:
        inNode = str(n) + "_in"
        outNode = str(n) + "_out"
        newNetwork.add_node(inNode)
        newNetwork.add_node(outNode)
        for inEdge in globalSubN.in_edges(n):
            if inEdge[0] in inNetwork.nodes():
                newNetwork.add_edge(inEdge[0], inNode) 
            elif inEdge[0] in outNetwork.nodes():
                newNetwork.add_edge(inEdge[0], outNode)
            else:
                sys.exit("Somethings reaaaaaaaaalllly realllly wrong. But it could be that reactions overlap? No....that's not possible since the d")
        for outEdge in globalSubN.out_edges(n):
            if outEdge[1] in inNetwork.nodes():
                newNetwork.add_edge(inNode, outEdge[1])
            elif outEdge[1] in outNetwork.nodes():
                newNetwork.add_edge(outNode, outEdge[1])
    newNetwork.remove_nodes_from(nodeDeletelist)
    inEdges = list(outNetwork.in_edges(startingNode))
    outEdges = list(inNetwork.out_edges(startingNode))
    newNetwork.remove_edges_from(outEdges)
    newNetwork.remove_edges_from(inEdges)
    return newNetwork
#############################
#############################


def processCircuitsAll(circuits, description:str):
    global parallelBool
    global equivClassLengthDict
    global cycleLengthDict
    global species
    global elementaryCircuits
    global fluffleBool
    #global allCircuitsPath
    breakBool = False
    n=0
    futureSet = set()
    circuitCounter=0
    if parallelBool==True:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            while True:
                try:
                    futureSet.add(executor.submit(analyzeElementaryCircuits, next(circuits)))
                    n+=1
                except StopIteration as sti:
                    breakBool = True
                if breakBool == True or n>1e7:
                    for f in tqdm(concurrent.futures.as_completed(futureSet), leave = False, total = n, desc= description+species):
                        try:
                            remove, circuit, eqClass = f.result()
                            if remove == False:
                                l = len(circuit)
                                cycleLengthDict[l]=cycleLengthDict.setdefault(l,0)+1
                                if fluffleBool==True:
                                    elementaryCircuits.append(c)
                                if checkEquivalenceClass(circuit, eqClass):
                                    circuitCounter+=1
                                    lequiv = len(eqClass)*2
                                    equivClassLengthDict[lequiv]=equivClassLengthDict.setdefault(lequiv,0)+1
                                    #cycleFile.write(str(circuit) + "\n")
                            del circuit, f
                        except Exception as exc:
                            print('%r generated an exception: %s', exc)
                    del futureSet
                    futureSet=set()
                    n=0
                    if breakBool==True:
                        break
    else:
        for c in circuits:
            remove, circuit, eqClass = analyzeElementaryCircuits(c)
            if remove == False:
                l = len(circuit)
                cycleLengthDict[l]=cycleLengthDict.setdefault(l,0)+1
                if fluffleBool==True:
                    elementaryCircuits.append(c)
                if checkEquivalenceClass(circuit, eqClass):
                    circuitCounter+=1
                    lequiv = len(eqClass)*2
                    equivClassLengthDict[lequiv]=equivClassLengthDict.setdefault(lequiv,0)+1
    del circuits
    gc.collect()
    return circuitCounter
#############################
#############################


def processCircuits(circuits, leaf:bool, left:bool, circuitCounter:int):
    global coreBool
    if leaf == True:
        description = "Analyzing elementary circuits for leaf for"
    else:
        if left == True:
            description = "Analyzing elementary circuits of left outnetwork for "
        else:
            description = "Analyzing elementary circuits of right outnetwork for "
    if coreBool == True:
        processCircuitsCore(circuits, description)
    else:
        circuitCounter += processCircuitsAll(circuits, description)
    return circuitCounter
#############################
#############################


def readArguments():
    inputBool = False
    xmlBool = False
    circuitBool = False
    checkNonMetzler = True
    threadBool = False
    equivClassBoundBool = False
    fluffleBool = False
    coreBool = False
    parallelBool = False
    speciesBool = False
    cycleDataBool = False
    for k in range(len(sys.argv)):
        newArgument = sys.argv[k]
        if newArgument == "-x" or newArgument=="--xmlFile":
            inputXMLFilePath = sys.argv[k+1]
            xmlBool = True
        elif newArgument == "-i" or newArgument=="--input":
            inputPickleFile = sys.argv[k+1]
            inputBool = True
        elif newArgument == "-b" or newArgument=="--circuitBound":
            circuitBound = int(sys.argv[k+1])
            circuitBool = True
        elif newArgument == "-n" or newArgument == "-nonMetzler":
            nonMetzlerString = sys.argv[k+1].lower()
            if nonMetzlerString == "false":
                checkNonMetzler=False
        elif newArgument == "-t" or newArgument == "--noThreads":
            noThreads = int(sys.argv[k+1])
            threadBool = True
        elif newArgument == "-e" or newArgument == "--equivClassBound":
            equviClassBound = int(sys.argv[k+1])
            equivClassBoundBool = True
        elif newArgument == "-f" or newArgument == "--fluffles":
            fluffleBool = True
        elif newArgument == "-c" or newArgument == "--cores":
            coreBool = True
        elif newArgument == "-p" or newArgument == "--parallel":
            parallelBool = True
        elif newArgument == "-s" or newArgument == "--Species":
            species = sys.argv[k+1]
            speciesBool = True
        elif newArgument == "-o" or newArgument == "--ouput":
            outputPath = sys.argv[k+1]
            cycleDataBool = True
    if inputBool==False:
        sys.exit("Please specify input pickle file.")
    if xmlBool == False:
        sys.exit("Please specify original xml-file.")    
    if speciesBool == False:
        sys.exit("Please specify species!")
    if circuitBool == False:
        circuitBound = 20
    if threadBool == False:
        noThreads = 2
    if equivClassBoundBool == False:
        equviClassBound = 20
    if cycleDataBool == False:
        outputPath="./cycleData/"
    return inputXMLFilePath, inputPickleFile, circuitBound, checkNonMetzler, noThreads, equviClassBound, fluffleBool, coreBool, parallelBool, species, outputPath
#############################
#############################


def writeDataToListsAndDicts(parameters, aM, nAM, nMAC, nMnAC):
    # 1. Metzler
    parameters["autocatalyticMetzlerUnstableCycles"] += aM
    parameters["nonAutocatalyticMetzlerCycles"] += nAM
    # 2. Non-Metzler
    parameters["notMetzlerAutocatalyticCycles"] += nMAC 
    parameters["nonMetzlerNonAutocatalyticCycles"] += nMnAC
    del aM, nAM, nMAC, nMnAC
    return
#############################
#############################


def writeDictionaryFromStoichiometricMatrix(S:np.matrix):
    SDict = {}
    for i in range(np.shape(S)[0]):
        for j in range(np.shape(S)[1]):
            SDict[(i,j)] = S[i,j]
    return SDict
#############################
#############################


def writeStoichiometricMatrixOutput(parameters:dict, path:str):
    S = parameters["StoichiometricMatrix"]
    with open(path, "w") as file:
        for i in range(sp.shape(S)[0]):
            for j in range(sp.shape(S)[1]):
                if j==0:
                    line = str(S[i,j])
                else:
                    line+= " " + str(S[i,j])
            if i == sp.shape(S)[0]-1:
                file.write(line)
            else:
                file.write(line+str("\n"))
    return
#############################
#############################

############################################################################################################################################################################## 
##############################################################################################################################################################################

#===========================================================================================================
#                          Functions to find only the cores in this setion 
#===========================================================================================================


def analyzeElementaryCircuitsCore(c:list):
    # Compute stochastic matrix and check metzler a
    mrEdgeSet = getEquivalenceClass(c)
    remove = determineDuplicates(c)
    unstable = False
    metzler = False
    if remove == False:
        subS, metzler = computeSubstochasticMatrixForSetOfMREdges(parameters, mrEdgeSet)
        if metzler == True:
            unstable = determineStability(subS)
    return remove, c, mrEdgeSet, unstable, metzler
#############################
#############################


def assembleCores(parameters:dict, Q:deque, E:dict, speedCores:set):
    cutoff = parameters["cutoffLargerCycles"]
    while True:
        if len(speedCores)>1e4:
            maxVal = min(int(2e12/len(Q)), len(Q))
            with concurrent.futures.ProcessPoolExecutor(max_workers=noThreads) as executor:
                for f in tqdm(concurrent.futures.as_completed(executor.submit(callAssembleCythonCores, Q[i], E[Q[i]], cutoff) for i in range(maxVal)), total=maxVal, leave = False):
                    Q.popleft()
                    try:
                        equivClass, newEquivClasses, change  = f.result()
                        #Subset relationships
                        for c, cValue in change:
                            eValue = E[c]
                            eValue["Predecessors"].update(cValue["Predecessors"])
                            if "Core" in cValue:
                                eValue["Core"] = cValue["Core"]
                            if "Leaf" in cValue:
                                eValue["Leaf"]= cValue["Leaf"]
                        for newEquiv, newValues in newEquivClasses.items():
                            newFrozen = frozenset(newEquiv)
                            if newFrozen in E:
                                eValue = E[newFrozen]
                                eValue
                                if newValues["Leaf"]==False:
                                    eValue["Leaf"]==False
                                if newValues["Core"]==False:
                                    eValue["Core"]==False
                                    speedCores.discard(newFrozen)
                            else:
                                E[newFrozen]=newValues
                                if newValues["Autocatalytic"]==False:
                                    if len(newEquiv)<cutoff:
                                        Q.append(newFrozen)
                                else:
                                    speedCores.add(newFrozen)
                        # New cores
                        
                    except Exception as exc:
                        print('%r generated an exception: %s', exc)
                        print(traceback.format_exc())
        else:
            equivClass = Q.popleft()
            intersecEquivClasses= getIntersectingEquivClassesCores(equivClass, E)
            for interEqCl in intersecEquivClasses:
                newEquivClass = equivClass | interEqCl
                if len(newEquivClass)<=cutoff:
                    newFrozen = frozenset(newEquivClass)
                    if newFrozen not in E:
                        equivClassValues = E[equivClass]
                        interEqClValues = E[interEqCl]
                        plausible, newMR = checkMatch(equivClassValues, interEqClValues)
                        if plausible:        
                            S,metzler = computeSubstochasticMatrixForSetOfMREdges(parameters, newEquivClass)
                            if metzler == True:        
                                unstable= determineStability(S)
                                if unstable==True:
                                    if E[equivClass]["Autocatalytic"] == True or E[interEqCl]["Autocatalytic"] == True:
                                        E[newFrozen] = {"MR":newMR, "RM": 0, "Predecessors": {equivClass, interEqCl}, "Leaf": False, "Autocatalytic": True, "Metzler": True, "Update": False, "Visited": True, "Core": False}
                                    else:
                                        E[newFrozen] = {"MR":newMR, "RM": 0, "Predecessors": {equivClass, interEqCl}, "Leaf": False, "Autocatalytic": True, "Metzler": True, "Update": False, "Visited": True, "Core": True}
                                        speedCores.add(newFrozen)
                                else:
                                    E[newFrozen] = {"MR":newMR, "RM": 0, "Predecessors": {equivClass, interEqCl}, "Leaf": False, "Autocatalytic": False, "Metzler": True, "Update": False, "Visited": True, "Core": False}
                                if len(newFrozen)<=cutoff:
                                    Q.append(newFrozen)
        if len(Q)==0:
            break
    return
#############################
############################# 


def callAssembleCythonCores(equivClass:frozenset, equivClassValues:dict, cutoff:int):
    global elemE, M, circuitIdMrEdgeDict, bigS, mID, rID
    newEquivClasses, change = assembleCythonCores(equivClass, equivClassValues, elemE, M, circuitIdMrEdgeDict, cutoff, bigS, mID, rID)
    return equivClass, newEquivClasses, change
#############################
#############################


def checkEquivalenceClassCore(c, eqClass, autocatalytic):
    """
    Classify elementary circuits into equivalence classes according their 
    vertices and edges.

    On invocation, this function determines information about the vertices 
    and edges of elementary circuits given on the list queue. Importantly,
    elementary circuits vertex and edge-sets (E and E1: only metabolite -> reaction edges) serve 
    as keys and elementary circuits are stored according to equivalence classes
    invoked by these properteis in different dictionaries.
    
    Parameters
    ----------
    Q : list
        Contains all elementary circuits enumerated by the Johnsons-Algorithm in 
        def(analysePartitionTree).

    E : dict 
        Key: (frozen sets) of equivalence classes composed of MR-edges
        Value: tuple(metabolites, reactions)

    M : dict
        Key: One MR-edge
        Value: Set of elementary circuits containing this particular MR-edge

    cycleIDDict : dict
        Key: Integer (subsequently termed cycle-identifier)
        Value: Elementary circuit
    
    circuitIdMrEdgeDict : dict
        Key: cycle-identifier
        Value: Set of MR-edges

    """
    # 0. Read Variables
    global E                                                                        # \mathcal{E} im paper
    global Q 
    global M                                                                        # Dictionary Key: One! MR-edge only, Value: Set of cycles containing this MR-edge
    global circuitIdDict                                                            # Dictionary Key: cycle-identifier (int), Value: Cycle
    global circuitIdMrEdgeDict                                                      # Dictionary Key: Cycle-Identifier Value: Set of MR-edges contained in this cycle    
    global speedCores
    # 1. Iterate over all cycles
    i = len(Q)
    # 2. Iterate only over MR-edges
    frozenEq = frozenset(eqClass)    
    if frozenEq not in E:                                           # If MR-edgeset has not been added to bump-equivalence classes of elem. circuits, add it to new Q 
        mr = {}
        rm = {}
        Q.append(frozenEq)
        for e in eqClass:
            M.setdefault(e, set()).add(i)                           # Add edge if not already done to mrEdgeCycleDict with value of the current elementary circuit
            mr[e[0]]=e[1]
            rm[e[1]]=e[0]
        circuitIdMrEdgeDict[i] = frozenEq                           # Remember for this particular elementary circuit, which     
        E[frozenEq] = {"MR": mr, "RM": rm, "Predecessors": set(), "Leaf": True, "Autocatalytic": autocatalytic, "Metzler": True, "Visited": True, "Core": autocatalytic}     # Add new equivalence class to E with corresponding dictionary for M-R and R-M relationship, a set for precursors, and four flags: autocatalytic, Metzler, leaf, visited
        circuitIdDict[i] = c
        if autocatalytic == True:
            speedCores.add(frozenEq)
        return True
    else:
        return False
#############################
#############################  

    return
#############################
#############################


def getIntersectingEquivClassesCores(equivClass:set, E:dict):
    '''Determine the those equivalence classes (sets of MR-edges) that intersect with the current cycle of interest.
    
    Parameters
    ----------

    equivClass : set
        Set of MR-edges of the current cycle to check 
    
        circuitIdMrEdgeDict : dictionary
            Key: cycle identifier (e.g. ckey)
            Value: set of metabolite-reaction edges 

        M : dictionary
            Key: frozenset of metabolite-reaction (MR) edges
            Value: Set of cycles corresponding exhibiting these MR-edges
    '''
    global M 
    global circuitIdMrEdgeDict
    intersecEquivClasses = []                                                           # Initiate empty set to 
    all_circuits = set(chain.from_iterable(M[e] for e in equivClass))
    equivClassValues = E[equivClass]
    autocatalytic = equivClassValues["Autocatalytic"]
    for c in all_circuits:            
        cEqCl = circuitIdMrEdgeDict[c]
        if cEqCl in equivClassValues["Predecessors"]:
            continue
        elif equivClass in E[cEqCl]["Predecessors"]:
            continue
        elif equivClass == cEqCl:
            continue
        elif equivClass < cEqCl:
            value = E.setdefault(cEqCl, {})
            value.setdefault("Predecessors", set()).add(equivClass)
            value["Leaf"] = False
            if equivClassValues["Autocatalytic"]==True:
                speedCores.discard(cEqCl)
                value["Core"]=False
        elif cEqCl < equivClass:
            equivClassValues.setdefault("Predecessors", set()).add(cEqCl)
            equivClassValues["Leaf"] = False
            if E[cEqCl]["Autocatalytic"]==True:
                speedCores.discard(equivClass)
                equivClassValues["Core"]=False
        else:
            if autocatalytic == True:
                continue
            elif E[cEqCl]["Autocatalytic"]==True:
                continue
            else:
                intersecEquivClasses.append(cEqCl)    
    return intersecEquivClasses
#############################
#############################


def processCircuitsCore(circuits, description:str):
    global parallelBool
    global noThreads
    global species
    circuitCounter=0
    n=0
    breakBool = False
    if parallelBool == True:
        with concurrent.futures.ProcessPoolExecutor(max_workers=noThreads) as executor:
            futureSet = set()
            while True:
                try:
                    futureSet.add(executor.submit(analyzeElementaryCircuitsCore, next(circuits)))
                    n+=1
                except StopIteration as sti:
                    break
                if n>1e4:
                    break
            Iter = concurrent.futures.as_completed(futureSet)
            sillCircuits = False
            #for f in tqdm(concurrent.futures.as_completed(futureSet), leave = False, total = n, desc= description+species):
            while True:
                try:
                    f = next(Iter)
                except StopAsyncIteration:
                    breakBool=True
                try:
                    remove, circuit, eqClass, autocatalytic, metzler = f.result()
                    if remove == False:
                        if metzler == True:
                            l = len(circuit)
                            cycleLengthDict[l]=cycleLengthDict.setdefault(l,0)+1
                            if checkEquivalenceClassCore(circuit, eqClass, autocatalytic):
                                circuitCounter+=1
                                lequiv = len(eqClass)*2
                                equivClassLengthDict[lequiv]=equivClassLengthDict.setdefault(lequiv,0)+1
                                #cycleFile.write(str(circuit) + "\n")
                    del circuit, f
                except Exception as exc:
                    print('%r generated an exception: %s', exc)
                if sillCircuits == False:
                    try:
                        futureSet.add(executor.submit(analyzeElementaryCircuitsCore, next(circuits)))
                        n+=1
                    except StopIteration as sti:
                        sillCircuits = True
                if breakBool==True:
                    break
    else:
        for c in circuits:
            remove, circuit, eqClass, autocatalytic, metzler = analyzeElementaryCircuitsCore(c)
            if remove == False:
                if metzler == True:
                    l = len(circuit)
                    cycleLengthDict[l]=cycleLengthDict.setdefault(l,0)+1
                    if checkEquivalenceClassCore(circuit, eqClass, autocatalytic):
                        circuitCounter+=1
                        lequiv = len(eqClass)*2
                        equivClassLengthDict[lequiv]=equivClassLengthDict.setdefault(lequiv,0)+1
    return circuitCounter
#############################
#############################


def checkCoreInclusion(c):
    global speedCores
    global noCore
    # Return immediately if c is flagged as no core
    coreFlag = True
    for d in speedCores:
        if d<c:
            coreFlag=False
            break
    return coreFlag, c
#############################
#############################


def checkUniquenessOfCores(cores):
    global noThreads
    realCores = set()
    global noCore
    print("Checking cores now for inclusion")
    if len(cores)>1e4:
        with concurrent.futures.ProcessPoolExecutor(max_workers=noThreads) as executor:
            for f in tqdm(concurrent.futures.as_completed(executor.submit(checkCoreInclusion, c) for c in cores), total=len(cores)):
                try:
                    coreFlag, c = f.result()
                    if coreFlag == True:
                        realCores.add(c)
                except Exception as exc:
                    print('%r generated an exception: %s', exc)
                    print(traceback.format_exc())
    else:
        for c in cores:
            coreFlag = True
            if c in noCore:
                continue
            for d in cores:
                if c==d:
                    continue
                elif d<c:
                    coreFlag=False
                    break
                elif c<d:                               # c<d
                    noCore.add(d)
            if coreFlag==True:
                realCores.add(c)
    return realCores
#############################
#############################



#############################
#############################
#===========================================================================================================
#                                       MAIN
#===========================================================================================================

timeStamp = time.time()
gc.enable()

# 1. Define variables
inputXMLFilePath, inputPickleFile, circuitBound, checkNonMetzler, noThreads, cutoffLargerCycles, fluffleBool, coreBool, parallelBool, species, cycleDataPath = readArguments()

reader = libsbml.SBMLReader()
document = reader.readSBML(inputXMLFilePath)
model = document.getModel()

with open (inputPickleFile, "rb") as pklFile:
    parameters, partitionTree, siblings, leaves, uRN, usefulNetwork = pickle.load(pklFile)

x = len(parameters["metabolites"])
r = len(parameters["reactions"])

cutoffLargerCycles = min(cutoffLargerCycles/2, (min(x, r)))
print("=====================================================================================================================")
print("=====================================================================================================================")
print("=====================================================================================================================")
print("The current cutoff in the size of MR-edges is", cutoffLargerCycles)
print("=====================================================================================================================")
print("=====================================================================================================================")
print("=====================================================================================================================")
parameters["cutoffLargerCycles"] = cutoffLargerCycles 
print(inputPickleFile)
print(parameters.keys())

parameters["noThreads"] = noThreads

parameters["mID"], parameters["rID"], parameters["iDM"], parameters["iDR"], parameters["mList"], parameters["rList"] = getIDDicts(parameters["metabolites"], parameters["reactions"])
parameters["StoichiometricMatrix"] = generateStoichiometricMatrix(parameters, model)
parameters["StoichiometricMatrixDict"] = writeDictionaryFromStoichiometricMatrix(parameters["StoichiometricMatrix"])
allCircuitsPath = "/scratch/richard/Autocatalysis/cycleData/"

parameters["coreBool"] = coreBool
mID = parameters["mID"]
rID = parameters["rID"]
bigS = parameters["StoichiometricMatrix"]
globalSubN = nx.DiGraph()
globalLeftChild = nx.DiGraph()
globalRightChild = nx.DiGraph()

cycleDict = {}
cycleLengthDict = {}
elementaryCircuits = []
equivClassLengthDict = {}
E = {}                                                                            # \mathcal{E} im paper
elemE = {}
Q = deque()                                                                       # Q im paper
M = {}                                                                            # M in the draft, Dictionary Key: One! MR-edge only, Value: Set of cycles containing this MR-edge
circuitIdDict = {}                                                                # Dictionary Key: cycle-identifier (int), Value: Cycle
circuitIdMrEdgeDict = {}                                                          # Dictionary Key: Cycle-Identifier Value: Set of MR-edges contained in this cycle   
speedCores = set()
noCore = set()
maxRAM = 0

if not os.path.exists(cycleDataPath):
    os.makedirs(cycleDataPath)
if not os.path.exists(cycleDataPath+species):
    os.makedirs(cycleDataPath+species)

treeCounter = int(inputPickleFile.split("partitionTree")[1].split(".pkl")[0])
# writeStoichiometricMatrixOutput(parameters, allCircuitsPath+species +"/"+"stoichiometricMatrix"+str(treeCounter)+".txt")
outputPickleFilePath = cycleDataPath + species + "/partitionTreeData" + str(treeCounter) + ".pkl"

file = open(cycleDataPath + species +"/allCycles"+ str(treeCounter) +".txt", "w")
file.close()
analysePartitionTree(parameters, partitionTree, siblings, leaves, uRN, usefulNetwork, circuitBound, species, treeCounter)
parameters["cycleDict"] = cycleDict 
parameters["cycleLengthDict"] = cycleLengthDict
totalTime = time.time()-timeStamp
parameters["TotalTime"] = totalTime
parameters["MaxRAM"]=maxRAM/(1024**3)

with open(outputPickleFilePath, "wb") as file:
    pickle.dump((parameters, partitionTree, siblings, leaves, uRN, usefulNetwork), file)

