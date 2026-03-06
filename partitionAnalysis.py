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

def analyzeCycles(G:nx.Graph, analyzeDict:dict, overlapDict:dict, childrensDict:dict, leaves:set, subN:nx.DiGraph, bound:int):
    ''' AnalyzeCycles

      Upon invocation this function analyzes the cycles of a given subnetwork (subN) of the metabolic network, depending on the position of the corresponding vertex in the partition tree. If it is a leaf, then all cycles are enumerated and stored in E. If it is not a leaf, then only the cycles that are necessary to separate the cycles of the left and right child are enumerated and stored in E. To this end, for each overlapping metabolite m two networks are generated, one that passes m from right to left and one that passes m from left to right. Only the generator of nx.simple_cycles is handed to processCircuits() to avoid memory issues. The function returns the number of enumerated elementary circuits for statistic purposes. 


    Parameters
    ----------

    1. Global

    :param globalSubN: Empty nx.DiGraph that is filled with life depending on the function. It functions as a place holder that allows to use a global variable (e.g. in parallelization). GlobalSubN is set to the directed subnetwork corresponding to the currently analyzed vertex in the partition tree. Required for enumeration of cycles.
    :type globalSubN: nx.DiGraph

    :param E: Dictionary designated to store all CS equivalence classes. Keys: frozensets of MR-edges, value: dictionary with different information and datastructures. As an example: {"MR": mr, "RM": rm, "Predecessors": set(), "Leaf": True, "Autocatalytic": False, "Metzler": True, "Visited": False, "Core": False}. mr and rm are again dictionaries specifying the correspondence between metabolites and reactions for metabolite-to-reaction and reaction-to-metabolite edges, respectively.
    :type E: dict

    :param treeCounter: Specifies the number of the analyzed pickle file (for documentation purposes). 
    :type treeCounter: int

    :param cycleDataPath: Specifies where the enumerated cycles should be stored (for documentation purposes). 
    :type cycleDataPath: str    

    2. Local    

    :param G: Currently analyzed vertex in the partition tree.
    
    :param species: Name of the species the metabolic network belongs to.
    :type species: str

    :param treeCounter: Specifies the number of the analyzed pickle file (for documentation purposes). 
    :type treeCounter: int

    :param cycleDataPath: Specifies where 
    :type cycleDataPath: str

    :param bound: Maximum length of enumerated elementary circuits.
    :type bound: int

    :param analyzeDict: Dictionary encoding if the cycles of the corresponding vertex in the partition tree have to be analyzed or not. Key: vertex in the partition tree, value: bool

    :param overlapDict: Dictionary encoding the overlap of metabolites between siblings in the partition tree. Key: vertex in the partition tree, value: set of overlapping metabolites of the corresponding children

    :param childrensDict: Dictionary encoding the left and right child of each vertex in the partition tree. Key: vertex in the partition tree, value: dictionary with keys "left" and "right" mapping to the corresponding subnetworks of the children.

    :param leaves: Set of vertices in the partition tree that have no outgoing edges, i.e. are not parents of any other vertex.
    :type leaves: set
    
    :paran subN: Directed subnetwork corresponding to the currently analyzed vertex in the partition tree.
    :type subN: nx.DiGraph

    :param bound: Maximum length of enumerated elementary circuits.
    :type bound: int

    Returns:
        - circuitCounter: Number of enumerated elementary circuits for statistic purposes.
    '''
    # Global variables
    # 0.1. Get global variables
    global globalSubN, E, species, treeCounter, cycleDataPath
    # 0.2 assign values
    globalSubN = subN                                                                                   # Set th global variable globalSubN to the currently analyzed subnetwork, required for parallelization
    circuitCounter = 0
    thousands = 1
    if analyzeDict[G] == True:                                                                          # So, if you have to analyze the network
        if G in leaves:                                                                                 # Determine if it is a leaf
            if len(E)>0:
                if int(len(E)/1000)>thousands:
                    print(len(E))
                    thousands+=1
            if len(E)>1e6:
                sys.exit("Size of elementary circuits getting too large, please reduce the size of the network.")
            leafSimpleCycles = nx.simple_cycles(subN, length_bound = bound)
            circuitCounter = processCircuits(circuits = leafSimpleCycles, leaf = True, left = False, circuitCounter = circuitCounter)
        else:                                                                                           # Otherwise, we need to make sure to now separate the cycles
            if len(E)>0:
                if int(len(E)/1000)>thousands:
                    print(len(E))
                    thousands+=1
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
            if len(E)>5e6*(12/noThreads):
                sys.exit("Size of elementary circuits getting too large, please reduce the size of the network.")
    return circuitCounter
#############################
#############################     


def analyzeElementaryCircuits(c:list):
    ''' AnalyzeElementaryCircuits
    
    Upon invocation this function analyzes the elementary circuit c by computing the corresponding MR-edges via valling getEquivalenceClass() and determining the duplicates via determineDuplicates(). The MR-edges are required for the generation of the CS matrix and the determination of the corresponding equivalence class. In particular, the CS equivalence relationship is defined by the MR-edges. Determine duplicateds identifies vertices that occur twice on an elementary circuit, which can only happend for inner vertices of the partition tree. For an example we refer to Fig. S6 of Golnik et al. (2026), "Enumeration of autocatalytic subsystems in large chemical reaction networks". If such a duplicated occurs remove is set to true and the circuit will be discard in the corresponding calling function analyzePartitionTree(). The function additionall returns the MR-edges and the duplicates for later use.

    Parameters
    ----------  
    :param c: List representing the currently analyzed elementary circuit as a list of vertices.

    Returns:
        - remove: Boolean specifying if the circuit should be removed due to duplicates.
        - c: List representing the currently analyzed elementary circuit as a list of vertices.
        - mrEdgeSet: Set of MR-edges corresponding to the currently analyzed elementary circuit.
    '''
    # Compute stochastic matrix and check metzler a
    mrEdgeSet = getEquivalenceClass(c)
    remove = determineDuplicates(c)
    return remove, c, mrEdgeSet
#############################
#############################


def analysePartitionTree(partitionTree:nx.DiGraph, siblings:dict, leaves:set, uRN:nx.Graph, network:nx.DiGraph, bound:int):
    ''' AnalzyePartitionTree
    
    Upon invocation this function is the main function parsing the partition tree that has been created in partitionNetwork and 
    represents the modularization of the metabolic network. However, each vertex in the partition tree represents a undirected reaction network 
    and needs to be translated into the corresponding directed metabolic network, which is achieved by invocation of the function generated subnetwork. All leaves are flagged visited and added to the qeueue current. Starting then with the leaves, the elementary circuits of the subnetwork are analyzed by invocation of the function analyzeCycles(). This is, however, executed only if the sibling of the currently analyed vertex was already visited. In this case, both siblings are analyzed and their parent vertex added to the queue current and flagged as visited. The childrens dictionary remembers the left and right children for overlapping metabolites. The overlap dictionary stores the overlap of metabolites between siblings for later analysis. After the traversal of the tree, all elementary circuits are stored in E and elemE, as well as in the list elementaryCircuits. Then, larger fluffle equivalence classes are assembled by invocation of assembleLargerEquivClassesParallel() and checked for autocatalycity by invocation of checkAutocatalycity(). If desired, also fluffle equivalence classes are assembled by invocation of assembleFluffles(). If the core flag is set only autocatalytic cores are enumerated. Finally, all results are written into the parameters dictionary for later use. 
    
    Parameters
    ----------
    
    1. Global

    :param sCC: Represents a strongly connected component of the input metabolic network after removal of unneccessary metabolites.
    :type sCC: nx.DiGraph

    :param parameters: Central dictionary storing multiple datastructures to avoid the massive transfer of datastructures to different subfunctions.
    :type parameters: dict

    :param E: Dictionary designated to store all CS equivalence classes. Keys: frozensets of MR-edges, value: dictionary with different information and datastructures. As an example: {"MR": mr, "RM": rm, "Predecessors": set(), "Leaf": True, "Autocatalytic": False, "Metzler": True, "Visited": False, "Core": False}. mr and rm are again dictionaries specifying the correspondence between metabolites and reactions for metabolite-to-reaction and reaction-to-metabolite edges, respectively.
    :type E: dict
            
    :param elemE: Same as E but only for CS-equivalence classes corresponding to elementary circuits. 
    :type elemE: dict

    :param Q: Collection that collects all CS equivalence classes that are designated for the assembly of larger fluffles in FIFO order.
    :type Q: deque

    :param cycleLengthDict: Dictionary storing the number of elementary circuits with a certain length. Only for later statistic reasons. Key: int, value: int
    :type cycleLength: dict
    
    :param elementaryCircuits: List storing all elementary circuits as lists provided by nx.simple_cycles(G). Required for enumeration of fluffle equivalence classes, if desired.
    :type elementaryCircuits: List

    :param equivClassLengthDict: Dictionary storing the number of CS equivalence classes of a certain length. Only for later statistic reasons. Key: int, value: int
    :type equivClassLengthDict: dict

    :param fluffleBool: Specifies if next to CS-equivalence classes also fluffle equivalence classes should be enumerated (not recommended due to complexity, only for small networks and academic reasons)
    :type fluffleBool: bool

    :param coreBool: Specifies if only autocatalytic cores or all autocatalytic CS-equivalence classes with CS matrices with irreducible Metzler part should be enumerated. 
    :type coreBool: bool

    :param speedCores: List for collecting autocatalytic cores if only cores are enumerated.
    :type speedCores: list

    2. Local

    :param partitionTree: Encodes the modularization of the given metabolic network that was created by partitionNetwork in a directed binary tree. Beginning with leaves, modules are analyzed separately before fused with their siblings and analyzed subsequently. 
    :type partitionTree: nx.DiGraph

    :param siblings: Dictionary encoding the affiliation of vertices in the partitionTree to their brother/sister vertices. In particular, each key maps to the second child of its parent.
    :type siblings: dict

    :param leaves: Set of vertices of the partition Tree that have not outoing edges.
    :type leaves: set

    :param uRN: Graph encoding the underlying undirected graph of the reaction network corresponding to the currently analzyed strongly connected component of the metabolic network (network) after reduction by the list of small metabolites.
    :typ uRN: nx.Graph

    :param network: Currently analzyed strongly connected componented of the input metabolic network. 
    :type network: nx.DiGraph

    :param bound: Maximum length of enumerated elementary circuits.
    :type bound: int

    :param uRN: Graph encoding the underlying undirected graph of the reaction network corresponding to the currently analzyed strongly connected component of the metabolic network (network) after reduction by the list of small metabolites.
    :typ uRN: nx.Graph

    Returns:
      - None
    '''
    
    # 0. Read global variables
    global parameters, E,  elemE, Q, cycleLengthDict, elementaryCircuits, equivClassLengthDict, fluffleBool, coreBool, speedCores 
    # 1. Define variables
    # 1.1 Dictionaries
    analyzeDict = {}                                                                # Dictionaries for storing information if an analysis on this node is necessary
    overlapDict = {}                                                                # Dictionary to store overlap of metabolites between to siblings
    childrensDict = {}                                                              # Dictionary to store all the cycles for subnetworks to impede enumerating them de nove
    # 1.1 Integers
    cycleCounter = 0                                                                # Counts number of enumerated elementary circuits
    # 1.2 Complexer datastructures 
    defineNewVariablesForParametersDictionary(parameters)                           # Assigns new variables to parameters dictionary for storage of the elementary circuits with different CS Matrices
    # 2. Initialize datastructures for traversal of partition Tree
    for l in leaves: 
        analyzeDict[l] = True                       # Each leaf has to be analysed
        overlapDict[l] = set()                      # Each overlap of metabolites for laeves is 0
    current = list(copy(leaves))                    # Start with leaves
    visited = copy(leaves)                          # Store what has already been visited
    # 3. Traverse Tree
    while True:
        subG  = current.pop(0)                      # Get first element from the queue 
        rootBool= False                             # Set root - boolean to false
        if subG == uRN:                             # If the current subnetwork is the complete undirected RN then the rootbool is True
            rootBool = True
        else:
            subGSibling = siblings[subG]                # Read sibling of this element
            if subGSibling not in visited:              # If the sibling is not visited yet, then we cannot deal with this problem now, 
                current.append(subG)                    # Move current element to the end of the queue (we'll deal with it later) and go to the next one
                continue
        if rootBool == True:
            print("Now checking the root vertex, might take some time")
        subnetwork, metabolites = generateSubnetwork(subG, network)                                  # Generate subnetwork
        # First analyze subG
        additionalCycles = analyzeCycles(subG, analyzeDict, overlapDict, childrensDict, leaves, subnetwork, bound)
        # 2. Non-Metzler
        cycleCounter+=additionalCycles
        if rootBool == False:                       # Root boolean tells us, when we have arrived at the root....then we don't have to look for siblings
            current.remove(subGSibling)             # Since we are dealing now with the current element and its sibling, remove the sibling from the queue, otherwise we would do the same thin twice
            siblingSubnetwork, siblingMetabolites = generateSubnetwork(subGSibling, network)      # Generate siblings subnetwork
            metaboliteOverlap = set(metabolites).intersection(set(siblingMetabolites))                              # Compute overlap of metabolites between the siblings networks for the parent
            for inEdge in partitionTree.in_edges(subG):                                                             # Determine if this is the root or not
                p = inEdge[0]
                childrensDict[p] = {}
                childrensDict[p]["left"] = subnetwork                               # Rememever the children
                childrensDict[p]["right"] = siblingSubnetwork
                overlapDict[p] = metaboliteOverlap
                if len(metaboliteOverlap)>1:
                    analyzeDict[p] = True
                else:
                    analyzeDict[p] = False
                current.append(p)
                visited.add(p)
            # Second analyze subG Sibling
            additionalCycles = analyzeCycles(subGSibling, analyzeDict, overlapDict, childrensDict, leaves, siblingSubnetwork, bound)
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


def assembleLargerEquivClassesParallel(parameters:dict, Q: deque, E:dict, equivClassLengthDict:dict):
    ''' assmbleLargerEquivClassesParallel

        Upon invocation this function assembles larger CS equivalence classes by invoking the cython function assembleCython() in parallel for all equivalence classes in Q. The cython function is required to speed up the assembly of larger CS equivalence classes, which is the most time consuming part of the algorithm. The function updates E and equivClassLengthDict with the new equivalence classes and the changes in already existing equivalence classes, which are returned by callAssembleCython(). In case the length of Q exceeds 1e3, the assembly is executed in parallel, otherwise it is executed sequentially to avoid the overhead of parallelization. The function returns nothing but updates E and equivClassLengthDict in place.

        Parameters 
        ----------
        :param parameters: Central dictionary storing multiple datastructures to avoid the massive transfer of datastructures to different subfunctions.
        :type parameters: dict  

        :param Q: Collection that stores all CS equivalence classes that are designated for the assembly of larger CS equivalence classes. Is processed within this function FIFO order.
        :type Q: deque

        :param E: Dictionary designated to store all CS equivalence classes. Keys: frozensets of MR-edges, value: dictionary with different information and datastructures. As an example: {"MR": mr, "RM": rm, "Predecessors": set(), "Leaf": True, "Autocatalytic": False, "Metzler": True, "Visited": False, "Core": False}. mr and rm are again dictionaries specifying the correspondence between metabolites and reactions for metabolite-to-reaction and reaction-to-metabolite edges, respectively.
        :type E: dict

        :equivClassLengthDict: Dictionary storing the number of CS equivalence classes of a certain length. Only for later statistic reasons. Key: int, value: int
        :type equivClassLengthDict: dict

        Returns:
            - None, but updates E and equivClassLengthDict in place.
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
            if sys.platform.startswith("linux"):
                executor = concurrent.futures.ProcessPoolExecutor(max_workers=noThreads)
            elif sys.platform == "darwin":
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=noThreads)
            else:
                executor = concurrent.futures.ProcessPoolExecutor(max_workers=noThreads)
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
            executor.shutdown()
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
    '''assmebleFluffles
    
    Upon invocation this function assembles larger fluffle equivalence classes by assembling all pairs of elementary circuits that have an overlap in their MR-edges and checking the plausibility of the matching by invocation of checkPlausibilityOfMatching(). The function updates the global variables cycleLengthDict, elementaryCircuits, cycleIDDict, cycleIDEdgeDict, cycleIDNodeDict with the new equivalence classes and the changes in already existing equivalence classes. The function returns the number of assembled larger fluffle equivalence classes.
    
    Parameters
    ----------
    1. Global 
    
    :param elementaryCircuits: List storing all elementary circuits as lists provided by nx.simple_cycles(G). 
    :type elementaryCircuits: List

    2. Local

    :param parameters: Central dictionary storing multiple datastructures to avoid the massive transfer of datastructures to different subfunctions.
    :type parameters: dict

    :queue: List of elementary circuits that are designated for the assembly of larger fluffle equivalence classes.
    :type queue: list

    Returns:
        - additionalCycleCounter: Integer counting the number of assembled larger fluffle equivalence classes.
    '''
    
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


def callAssembleCython(equivClass:frozenset, equivClassValues:dict, cutoff:int):
    ''' CallAssembleCython

    Upon invocation this function calls the cython function assembleCython() to assemble larger CS equivalence classes. The cython function is required to speed up the assembly of larger CS equivalence classes, which is the most time consuming part of the algorithm. The function returns the new equivalence classes and a dicitonary that records the changes in already existing equivalence classes for later use in assembleLargerEquivClassesParallel().

    Parameters
    ----------

    1. Global 

    :param elemE: Storing CS equivalence classes for elementary circuits as keys with additional information on the equivalenc class in a dictionary as value. 
    :type elemE: dict

    :param M: Dictionary storing the CS equivlance classes containing a particular MR-edge. Key: One single MR-edge, value: Set of CS equivalence classes containing this MR-edge. 
    :type M: dict

    :param circuitIdMrEdgeDict: Dictionary storing the MR-edges corresponding to a certain elementary circuit. Key: int (elementary circuit identifier), value: frozenset of MR-edges of the key.
    :type circuitIdMrEdgeDict: dict

    2. Local

    :param equivClass: Frozenset representing the currently analyzed equivalence class.
    :type equivClass: frozenset

    :param equivClassValues: Dictionary storing the information on the currently analyzed equivalence class. Key: "MR", "RM", "Predecessors", "Leaf", "Autocatalytic", "Metzler", "Update", "Visited", "Core", values: corresponding values
    :type equivClassValues: dict

    :param cutoff: Maximum length of CS equivalence classes that are assembled.
    :type cutoff: int

    Returns:
        - newEquivClasses: Dictionary storing the new equivalence classes that are assembled from the currently analyzed equivalence class and all overlapping CS equivalence classes. Keys: frozensets of MR-edges, value: dictionary with different information and datastructures. As an example: {"MR": mr, "RM": rm, "Predecessors": set(), "Leaf": True, "Autocatalytic": False, "Metzler": True, "Update": False, "Visited": False, "Core": False}. mr and rm are again dictionaries specifying the correspondence between metabolites and reactions for metabolite-to-reaction and reaction-to-metabolite edges, respectively.
        - change: Dictionary storing the changes in already existing equivalence classes for later use in assembleLargerEquivClassesParallel(). Keys: frozensets of MR-edges, value: Dictionary with different information and datastructures that are updated by the assembly of the currently analyzed equivalence class with all overlapping CS equivalence classes. 
    '''
    global elemE, M, circuitIdMrEdgeDict
    newEquivClasses, change = assembleCython(equivClass, equivClassValues,  elemE, M, circuitIdMrEdgeDict, cutoff)
    return newEquivClasses, change
#############################
#############################


def checkAutocatalycityRecursive(E:dict, equivClass:frozenset, equivClassProperties:dict, cores:set):
    '''checkAutocatalycityRecursive
    
    Upon invocation this function checks recursively if a given CS equivalence class is autocatalytic by checking if it has a Hurwitz-unstable Metzler matrix or if it has at least one predecessor that is autocatalytic. The function updates the global variable E with the information on the autocatalycity of the currently analyzed equivalence class and its predecessors. The function returns nothing but updates E in place. It is the recursive pendant of checkAutocatalycity() and does the actual checking of autocatalycity for each equivalence class in E. The function is designed to be called after the assembly of larger CS equivalence classes to check the autocatalycity of these larger CS equivalence classes.
    
    Parameters
    ----------
    :param E: Dictionary designated to store all CS equivalence classes. Keys: frozensets of MR-edges, value: dictionary with different information and datastructures. As an example: {"MR": mr, "RM": rm, "Predecessors": set(), "Leaf": True, "Autocatalytic": False, "Metzler": True, "Visited": False, "Core": False}. mr and rm are again dictionaries specifying the correspondence between metabolites and reactions for metabolite-to-reaction and reaction-to-metabolite edges, respectively.
    :type E: dict

    :param equivClass: Frozenset representing the currently analyzed equivalence class.
    :type equivClass: frozenset

    :param equivClassProperties: Dictionary storing the information on the currently analyzed equivalence class. Key: "MR", "RM", "Predecessors", "Leaf", "Autocatalytic", "Metzler", "Update", "Visited", "Core", values: corresponding values
    :type equivClassProperties: dict    

    :param cores: Set storing all autocatalytic cores, which are equivalence classes that are autocatalytic but no susbet of the set of MR-edges is an autocatalytic CS equivalence class.
    :type cores: set
    
    Returns:
        - None, but updates E in place with the information on the autocatalycity of the currently analyzed equivalence class and its predecessors.
    '''
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
    ''' checkAutocatalycity

    Upon invocation this function checks if the CS equivalence classes in E are autocatalytic by checking if they have a Hurwitz-unstable Metzler matrix or if they have at least one predecessor that is autocatalytic. The function updates the global variable E with the information on the autocatalycity of the currently analyzed equivalence class and its predecessors. The function returns a set of all autocatalytic cores, which are equivalence classes that are autocatalytic but no susbet of the set of MR-edges is an autocatalytic CS equivalence class. I calls checkAutocatalycityRecursive() which is the recursive counterpart of this function and does the actual checking of autocatalycity for each equivalence class in E. The function is designed to be called after the assembly of larger CS equivalence classes to check the autocatalycity of these larger CS equivalence classes and the ones corresponding to elementary circuits.

    Parameters
    ----------

    1. Global   

    :param checkNonMetzler: Boolean that is set to True if the user wants to enumerated also non-Metzler matrices and check for their autocatalytic capacity, which is computationally more expensive since also all CS equivalence classes corresponding to non-Metzler matrices are considered for the assembly of larger CS-equivalence classes. In this function, this only impacts checking the autocatalytic capacity of non-Metzler matrices.

    :param E: Dictionary designated to store all CS equivalence classes. Keys: frozensets of MR-edges, value: dictionary with different information and datastructures. As an example: {"MR": mr, "RM": rm, "Predecessors": set(), "Leaf": True, "Autocatalytic": False, "Metzler": True, "Visited  ": False, "Core": False}. mr and rm are again dictionaries specifying the correspondence between metabolites and reactions for metabolite-to-reaction and reaction-to-metabolite edges, respectively.
    :type E: dict

    Returns:
        - cores: Set of all autocatalytic cores, which are equivalence classes that are autocatalytic but no susbet of the set of MR-edges is an autocatalytic CS equivalence class.
    '''
    # 0 .Read global variables
    global checkNonMetzler

    # 1. Define new Variables
    cores = set()

    # 2. Start iterating over potential autocatalytic core CS equivalence classes. 
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
    """ checkEquivalenceClass
    
    Upon invocation, this function determines information about the vertices 
    and edges of elementary circuits given on the list queue. Importantly,
    elementary circuits vertex and edge-sets (E and E1: only metabolite -> reaction edges) serve 
    as keys and elementary circuits are stored according to equivalence classes
    invoked by these properteis in different dictionaries.
    
    Parameters
    ----------
    
    1. Global
    
        :param E: Dictionary designated to store all CS equivalence classes. Keys: frozensets of MR-edges, value: dictionary with different information and datastructures. As an example: {"MR": mr, "RM": rm, "Predecessors": set(), "Leaf": True, "Autocatalytic": False, "Metzler": True, "Visited": False, "Core": False}. mr and rm are again dictionaries specifying the correspondence between metabolites and reactions for metabolite-to-reaction and reaction-to-metabolite edges, respectively.
        :type E : dict 

        :param Q: Contains all elementary circuits enumerated by the Johnsons-Algorithm from def(analysePartitionTree).
        :type Q: list
    
        :param M: Dictionary storing the CS equivlance classes containing a particular MR-edge. Key: One single MR-edge, value: Set of CS equivalence classes containing this MR-edge. 
        :type M: dict  

        :param cycleIDDict: Dictionary mapping cycle identifiers to elementary circuits. Key: int (cycle identifier), value: elementary circuit
        :type cycleIDDict: dict

        :param circuitIdMrEdgeDict: Dictionary mapping cycle identifiers to the set of MR-edges contained in the corresponding elementary circuit. Key: int (cycle identifier), value: frozenset of MR-edges contained in the corresponding elementary circuit
        :type circuitIdMrEdgeDict: dict

    2. Local

        :param c: List of vertices representing an elementary circuit.
        :type c: list

        :param eqClass: Set of MR-edges contained in the elementary circuit represented by c.
        :type eqClass: set
    
    Returns:
        - Boolean value indicating whether the equivalence class of the currently analyzed elementary circuit has already been added to the global variable E or not. If it has not been added, the function adds the equivalence class to E and updates the global variables M, circuitIdDict, circuitIdMrEdgeDict with the information on the currently analyzed elementary circuit and its equivalence class. If it has already been added, the function returns False and does not update any global variable.
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
    ''' checkPlausibilityOfMatching
    
    Upon invocation this function checks the plausibility of the matching of a fluffle aned an elementary circuit by checking if each connected component of the intersection of the two graphs is an MR-path. The function returns a boolean value indicating whether the matching of the two cycles is plausible or not. The function is designed to be called in assembleFluffles() to check the plausibility of the matching of two cycles before assembling them to a larger fluffle equivalence class.

    Parameters
    ----------
    :param parameters: Central dictionary storing multiple datastructures to avoid the massive transfer of datastructures to different subfunctions.
    :type parameters: dict  

    :param c1Key: Integer representing the identifier of the first cycle, which is an elementary circuit or a fluffle equivalence class.
    :type c1Key: int    

    :param c2Key: Integer representing the identifier of the second cycle, which is an elementary circuit or a fluffle equivalence class.   
    :type c2Key: int

    :param cycleIDEdgeDict: Dictionary mapping cycle identifiers to the set of edges contained in the corresponding elementary circuit or fluffle equivalence class. Key: int (cycle identifier), value: frozenset of edges contained in the corresponding elementary circuit or fluffle equivalence class
    :type cycleIDEdgeDict: dict 

    :param cycleIDNodeDict: Dictionary mapping cycle identifiers to the set of nodes contained in the corresponding elementary circuit or fluffle equivalence class. Key: int (cycle identifier), value: frozenset of nodes contained in the corresponding elementary circuit or fluffle equivalence class
    :type cycleIDNodeDict: dict

    Returns:
        - plausible: Boolean value indicating whether the matching of the two cycles is plausible or not.
    '''
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
    ''' computeSubstochasticMatrixForSetOfMREdges
    
        Upon invocation this function determines the k x k CS matrix for a set of MR-edges, 
        i. e. a submatrix of the stochastic matrix S, where columns are re-ordered according 
        to the given MR relationship, which defines a perfect matching thus a CS. Accordingly, 
        for an edge (m,r) then r represents the i-th column if and only if m represents the i-th 
        row.

        Parameters
        ----------
        
        :param parameters: Central dictionary storing multiple datastructures to avoid the massive transfer of datastructures to different subfunctions.
        :type parameters: dict  

        :param newEquivClass: Set of MR-edges representing the equivalence class for which the CS matrix is to be computed.
        :type newEquivClass: set
        
        Returns:
            - subS: k x k CS matrix for the given set of MR-edges, where k is the number of MR-edges in the given set. The columns of subS are re-ordered according to the given MR relationship, which defines a perfect matching thus a CS. Accordingly, for an edge (m,r) then r represents the i-th column if and only if m represents the i-th row.
            - metzler: Boolean value indicating whether the computed CS matrix is a Metzler matrix or not. A Metzler matrix is a matrix where all off-diagonal entries are non-negative. 
            
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
    '''determineDuplicates
    
        Upon invocation this function checks if a given circuit contains duplicate vertices, which is the case if an intersecting metabolite that was split into a left and right one
        appears twice on the elementary circuit. In this case the circuit is discarded. It is a rare case but can occur for inner vertices of the partition tree, but not for the first intersecting metabolite, only for the second, third, etc. 

        Parameters
        ----------
        :param cycle: List of vertices representing an elementary circuit.
        :type cycle: list

        Returns:
            - Boolean value indicating whether the given circuit contains duplicate vertices or not. If it contains duplicate vertices, the function returns True, otherwise it returns False.
    '''
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
    '''defineNewVariablesForParametersDictionary
    
        Upon invocation this functions simply defines new variables in the parameters dictionary to store the identified elementary circuits in it. It defines the following variables in the parameters dictionary:

        1. Dictionaries:
        - detAutocatalysis: All autocatalytic CS equivalence classes with corresponding Metzler matrices that have the same absolute value of determinant are stored in a common value. Key: int (determinant), value list of tuples of CS equivalence classes + additional information

        2. Lists:
        - autocatalyticMetzlerUnstableCycles: All autocatalytic CS equivalence classes with corresponding Metzler matrices that are Hurwitz-unstable are stored in this list.
        - autocatalyticMetzlerUnstableInvertibleCycles: All autocatalytic CS equivalence classes with corresponding Metzler matrices that are Hurwitz-unstable and invertible are stored in this list.
        - autocatalyticMetzlerZeroDeterminantCycles: All autocatalytic CS equivalence classes with corresponding Metzler matrices that have determinant zero are stored in this list.
        - autocatalyticMetzlerZeroDeterminantUnstableCycles: All autocatalytic CS equivalence classes with corresponding Metzler matrices that have determinant zero and are Hurwitz-unstable are stored in this list.
        - autocatalyticMetzlerZeroDeterminantNotUnstableCycles: All autocatalytic CS equivalence classes with corresponding Metzler matrices that have determinant zero and are not Hurwitz-unstable are stored in this list.
        - nonAutocatalyticMetzlerCycles: All non-autocatalytic CS equivalence classes with corresponding Metzler matrices are stored in this list.
        - notMetzlerAutocatalyticCycles: All autocatalytic CS equivalence classes with corresponding non-Metzler matrices are stored in this list.
        - notMetzlerUnstableAutocatalyticCycles: All autocatalytic CS equivalence classes with corresponding non-Metzler matrices that are Hurwitz-unstable are stored in this list.
        - notMetzlerNotUnstableAutocatalyticCycles: All autocatalytic CS equivalence classes with corresponding non-Metzler matrices that are not Hurwitz-unstable are stored in this list.
        - notMetzlerNotStableNotUnstableAutocatalyticCycles: All autocatalytic CS equivalence classes with corresponding non-Metzler matrices that are not Hurwitz-unstable but also not Hurwitz-stable are stored in this list.    
        - notMetzlerStableAutocatalyticCycles: All autocatalytic CS equivalence classes with corresponding non-Metzler matrices that are Hurwitz-stable are stored in this list.
        - nonMetzlerNonAutocatalyticCycles: All non-autocatalytic CS equivalence classes with corresponding non-Metzler matrices are stored in this list.
        
        Parameters
        ----------
        :param parameters: Central dictionary storing multiple datastructures to avoid the massive transfer of datastructures to different subfunctions.
        :type parameters: dict  

        Returns:
            - None, but updates the parameters dictionary with new variables to store the identified elementary circuits in it.
        '''
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
    '''determineAutocatalycityNonMetzler
    
        Upon invocation this function checks if a given CS matrix that is not a Metzler matrix is autocatalytic by checking if it has a positive real eigenvalue. Since the matrix is not a Metzler matrix, we cannot use the Perron-Frobenius theorem to check for the existence of a positive real eigenvalue, which is why we have to use linear programming to check if there exists a positive vector v such that Sv>0, which is equivalent to the existence of a positive real eigenvalue. The function returns a boolean value indicating whether the given CS matrix that is not a Metzler matrix is autocatalytic or not.
        
        Parameters
        ----------
        :param S: k x k CS matrix that is not a Metzler matrix, where k is the number of MR-edges in the given set of MR-edges representing the CS equivalence class for which the autocatalycity is to be checked. The columns of S are re-ordered according to the given MR relationship, which defines a perfect matching thus a CS. Accordingly, for an edge (m,r) then r represents the i-th column if and only if m represents the i-th row.
        :type S: np.matrix  
        
        Returns:
            - autocatalytic: Boolean value indicating whether the given CS matrix that is not a Metzler matrix is autocatalytic or not.'''
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
    '''determineCircuitPropertiesMetzler
    
        This function determines the spectral properties of the Metzler CS matrix of a CS equivalence class by calling determineStability().
        
        Parameters
        ----------
        :param c: A set containing the MR edges of a CS equivalence class.
        :type c: set

        :param S: The k x k Metzler CS matrix, corresponding to the CS equivalence class c. k is the number of MR-edges.
        :type S: np.matrix
        
        Returns:
            - aM: List of autocatalytic Metzler cycles.
            - nAM: List of non-autocatalytic Metzler cycles.
            - unstable: Boolean value indicating whether the circuit is unstable.
    '''
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
    '''determineCircuitPropertiesNonMetzler
    
        This function determines the spectral properties of the non-Metzler CS matrix of a CS equivalence class by calling determineAutocatalycityNonMetzler().
        
        Parameters
        ----------
        :param c: A set containing the MR edges of a CS equivalence class.
        :type c: set

        :param S: The k x k Non-Metzler CS matrix, corresponding to the CS equivalence class c. k is the number of MR-edges.
        :type S: np.matrix
        
        Returns:
            - nMAC: List of non-autocatalytic Non-Metzler cycles.
            - nMnAC: List of non-autocatalytic Non-Metzler cycles.
            - autocatalytic: Boolean value indicating whether the given CS matrix that is not a Metzler matrix is autocatalytic or not.
    '''
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
    '''determineStability
    
        This function determines the stability of a CS matrix by checking the real parts of its eigenvalues.
        
        Parameters
        ----------
        :param T: The k x k CS matrix for which stability is to be checked. k is the number of MR-edges.
        :type T: np.matrix
        
        Returns:
            - unstable: Boolean value indicating whether the circuit is unstable.
    '''
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
    '''determineTypeOfAutocatalysis
    
        Upon invocation this function determines the type of autocatalysis of the identified autocatalytic CS equivalence classes with corresponding Metzler matrices by checking their spectral properties and determinant. The function updates the parameters dictionary with the identified autocatalytic CS equivalence classes with corresponding Metzler matrices in the following lists:

        - autocatalyticMetzlerUnstableCycles: All autocatalytic CS equivalence classes with corresponding Metzler matrices that are Hurwitz-unstable are stored in this list.
        - autocatalyticMetzlerUnstableInvertibleCycles: All autocatalytic CS equivalence classes with corresponding Metzler matrices that are Hurwitz-unstable and invertible are stored in this list.
        - autocatalyticMetzlerZeroDeterminantCycles: All autocatalytic CS equivalence classes with corresponding Metzler matrices that have determinant zero are stored in this list.
        - autocatalyticMetzlerZeroDeterminantUnstableCycles: All autocatalytic CS equivalence classes with corresponding Metzler matrices that have determinant zero and are Hurwitz-unstable are stored in this list.
        - autocatalyticMetzlerZeroDeterminantNotUnstableCycles: All autocatalytic CS equivalence classes with corresponding Metzler matrices that have determinant zero and are not Hurwitz-unstable are stored in this list.

        Parameters
        ----------
        :param parameters: Central dictionary storing multiple datastructures to avoid the massive transfer of datastructures to different subfunctions.
        :type parameters: dict  

        Returns:
            - None, but updates the parameters dictionary with the identified autocatalytic CS equivalence classes with corresponding Metzler matrices in the following lists:
                - autocatalyticMetzlerUnstableCycles: All autocatalytic CS equivalence classes with corresponding Metzler matrices that are Hurwitz-unstable are stored in this list.
                - autocatalyticMetzlerUnstableInvertibleCycles: All autocatalytic CS equivalence classes with corresponding Metzler matrices that are Hurwitz-unstable and invertible are stored in this list.
                - autocatalyticMetzlerZeroDeterminantCycles: All autocatalytic CS equivalence classes with corresponding Metzler matrices that have determinant zero are stored in this list.
                - autocatalyticMetzlerZeroDeterminantUnstableCycles: All autocatalytic CS equivalence classes with corresponding Metzler matrices'''
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
    '''generateEdgeCycleDict
    Upon invocation this function generates the following datastructures to store information about the identified elementary circuits and their edges and nodes:
    - edgeCycleDict: Dictionary mapping edges to the set of identifiers of elementary circuits containing this edge. Key: edge, value: set of cycle identifiers of elementary circuits containing this edge.
    - cycleIDDict: Dictionary mapping cycle identifiers. Key: int (el. circuit identifier), value: cycle (list of vertices representing the cycle).
    - cycleIDEdgeDict: Dictionary mapping elementary circuit identifiers to the set of edges contained in the corresponding elementary circuit. Key: int (elementary circuit identifier), value: frozenset of edges contained in the corresponding elementary circuit.
    - cycleIDNodeDict: Dictionary mapping elementary circuit identifiers to the set of nodes contained in the corresponding elementary circuit. Key: int (elementary circuit identifier), value: frozenset of nodes contained in the corresponding elementary circuit.
    - visitedEdges: Set of frozensets of edges, where each frozenset of edges represents the set of edges contained in an elementary circuit. This set is used to check if a fluffle equivalence class has already been enumerated.

    Parameters
    ----------
    :param queue: List of elementary circuits, where each elementary circuit is represented as a list of vertices (metabolite, reaction, metabolite, reaction, ...).
    :type queue: list

    Returns:
        - edgeCycleDict
        - cycleIDDict
        - cycleIDEdgeDict
        - cycleIDNodeDict
        - visitedEdges
        as described above. 
    '''
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
    '''generateStoichiometricMatrix
    Upon invocation this function generates the stoichiometric matrix S for the given metabolic network and the given lists of metabolites and reactions. The function returns the generated stoichiometric matrix S.
    
    Parameters
    ----------  
    
    :param parameters: Central dictionary storing multiple datastructures to avoid the massive transfer of datastructures to different subfunctions.
    :type parameters: dict  
    
    :param model: libsbml Model object representing the metabolic network for which the stoichiometric matrix is to be generated.
    :type model: libsbml.Model 
    
    Returns:
        - S: Stoichiometric matrix for the given metabolic network
    '''
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


def generateSubnetwork(subG:dict, metabolicNetwork:nx.DiGraph):
    '''generateSubnetwork
        Upon invocation this function generates the subnetwork of the given metabolic network that is induced by the undirected reaction graph of the currently analyzed vertex of the partition tree. The function returns the generated subnetwork and the list of metabolites contained in the subnetwork.
        
        Parameters
        ----------
        :param subG: Undirected reaction graph of the currently analyzed vertex of the partition tree.
        :type subG: dict
        :param metabolicNetwork: The metabolic network for which to generate the subnetwork.
        :type metabolicNetwork: nx.DiGraph
        Returns:
            - subnetwork: The subnetwork of the given metabolic network that is induced by the undirected reaction graph of the currently analyzed vertex of the partition tree.
            - metabolites: List of metabolites contained in the generated subnetwork.
        '''
    
    metabolites = set()
    subGReactions = set(subG.nodes()) 
    for r in subGReactions:
        for inEdge in metabolicNetwork.in_edges(r):
            metabolites.add(inEdge[0])
        for outEdge in metabolicNetwork.out_edges(r):
            metabolites.add(outEdge[1])
    subnetwork = nx.subgraph(metabolicNetwork, metabolites.union(subGReactions)).copy()
    return subnetwork, sorted(list(metabolites))
#############################
#############################


def getIntersectingCycles(cKey:int, cycleIDEdgeDict:int, edgeCycleDict:dict):
    '''getIntersectingCycles
        Upon invocation this function determines the set of elementary circuits that intersect with the currently analyzed fluffle equivalence class by checking which of the edge-set intersections are not empty. The function returns the set of cycles that intersect with the currently analyzed cycle and the set of edges contained in the currently analyzed cycle.
        
        Parameters
        ----------

        :param cKey: Integer representing the identifier of the currently fluffle equivalence class.
        :type cKey: int 
        
        :param cycleIDEdgeDict: Dictionary mapping cycle identifiers to the set of edges contained in the corresponding elementary circuit or fluffle equivalence class. Key: int (cycle identifier), value: frozenset of edges contained in the corresponding elementary circuit or fluffle equivalence class
        :type cycleIDEdgeDict: dict
        :param edgeCycleDict: Dictionary mapping edges to the set of cycles they are contained in. Key: edge, value: set of cycle identifiers
        :type edgeCycleDict: dict
        
        Returns:
            - intersectingCycles: Set of cycles that intersect with the currently analyzed cycle.
            - edgesC: Set of edges contained in the currently analyzed cycle.
        '''
    edgesC = cycleIDEdgeDict[cKey]
    intersectingCycles = set()
    for e in edgesC:
        intersectingCycles = intersectingCycles.union(edgeCycleDict[e])
    return intersectingCycles, edgesC
#############################
#############################


def getEquivalenceClass(c:list):
    """ getEquivalenceClass
    
    Determine the set of MR-edges and thereby the CS equivalence class of a given elementary circuit c. and edges.
    
    Parameters
    ----------
    :param c: List of vertices representing an elementary circuit, where the vertices are ordered as follows: metabolite, reaction, metabolite, reaction.
    :type c: list

    Returns:
        - mrEdgeSet: Set of MR-edges contained in the given elementary circuit.
    """

    # 0. Define Variables
    global parameters
    mrEdgeSet = set()
    metabolicNetwork = parameters["metabolicNetwork"]  
    # 2. Iterate only over MR-edges
    offset=0
    if type(c[0])!=str:
        if metabolicNetwork.nodes[c[0]]["Type"] == "Reaction":
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
    ''' getIDDicts
    
        Upon invocation this function generates the following datastructures to store information about the metabolites and reactions of the given metabolic network:
            - mID: Dictionary mapping metabolite identifiers to integer indices. Key: metabolite identifier (int) from the metabolic network, value: integer index in the list of metabolites, thus the row index in the stoichiometric matrix S.
            - rID: Dictionary mapping reaction identifiers to integer indices. Key: reaction identifier (int) from the metabolic network, value: integer index in the list of reactions, thus the column index in the stoichiometric matrix S.
            - iDM: Dictionary mapping integer indices to metabolite identifiers. Key: integer index in the list of metabolites, thus the row index in the stoichiometric matrix S, value: metabolite identifier (int) from the metabolic network.
            - iDR: Dictionary mapping integer indices to reaction identifiers. Key: integer index in the list of reactions, thus the column index in the stoichiometric matrix S, value: reaction identifier (int) from the metabolic network.
            - mList: List of metabolite identifiers, where the i-th element of the list corresponds to the metabolite identifier that is mapped to the i-th row of the stoichiometric matrix S.
            - rList: List of reaction identifiers, where the j-th element of the list corresponds to the reaction identifier that is mapped to the j-th column of the stoichiometric matrix S.
        
        Parameters
        ----------
            :param metabolites: Set of metabolite identifiers (int) from the metabolic network.
            :type metabolites: set
            :param reactions: Set of reaction identifiers (int) from the metabolic network.
            :type reactions: set
        
        Returns:
            - mID: 
            - rID: 
            - iDM:
            - iDR:
            - mList:
            - rList:
            as described above.
        '''
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
    ''' getIntersectingEquivClassesParallel

        Upon invocation this function determines the equivalence classes (sets of MR-edges) that intersect with the current CS equivalence class handed to the function. It represents the version of getIntersectingEquivClasses() used in parallel processing of CS equivalence class assmebly.
    
        Parameters
        ----------

        1. Global
        
        :param M: Dictionary mapping MR-edges to the CS equivalence classes containing them. Key one MR-edge. Value: Set of CS equivalence classes containing this MR-edge. 
        :type M: dict

        :param circuitIdMrEdgeDict: Dictionary mapping elementary circuit identifiers to the set of MR-edges contained in the corresponding elementary circuit. Key: el.circuit  identifier (e.g. ckey), value: set of metabolite-reaction edges
        
        2. Local

        :param equivClass: Set of MR-edges of the current CS equivalence class
        :type equivClass: set

    
        Returns
            - intersecEquivClasses: List of CS equivalence classes that intersect with equivClass.
            - changeE: Dictionary that contains intersecting CS equivalence classes that have already been found and values that are going to be updated in the global E dictionary.
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
    ''' getIntersectingEquivClasses

        Upon invocation this function determines the equivalence classes (sets of MR-edges) that intersect with the current CS equivalence class handed to the function. In contrast to the parallelized version of this function, change are directly written into the global E dictionary, which is impossible in the parallelized version due to inconsistencies upon asynchronous writing and reading operations by multiple processes.
    
        Parameters
        ----------

        1. Global
        
        :param M: Dictionary mapping MR-edges to the CS equivalence classes containing them. Key one MR-edge. Value: Set of CS equivalence classes containing this MR-edge. 
        :type M: dict

        :param circuitIdMrEdgeDict: Dictionary mapping elementary circuit identifiers to the set of MR-edges contained in the corresponding elementary circuit. Key: el.circuit  identifier (e.g. ckey), value: set of metabolite-reaction edges
        
        2. Local

        :param equivClass: Set of MR-edges of the current CS equivalence class
        :type equivClass: set

        :param E: Dictionary mapping CS equivalence classes to their properties.
        :type E: dict
        
    
        Returns
            - intersecEquivClasses: List of CS equivalence classes that intersect with equivClass.
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
    '''getOutNetwork
    
        Upon invocation this function generates the left (right) out- and right (left) in-network of the currently analyzed vertex of the partition tree for the current intersecting metabolite. In this way, only elementary circuits passing the intersecting metabolite in one or the other directtion are enumerated and enumeration of already identified elementary circuits that are completely contained in on the left or right child is avoidewd. The function returns the generated orientied network.
        
        Parameters
        ----------
        
        1. Global
        :param globalSubN: Global variable that is the subnetwork of the currently analyzed vertex of the partition tree.

        :param outNetwork: The network in to which all outgoing edges from the intersecting metabolite point.
        :type outNetwork: nx.DiGraph

        :param inNetwork: The network from which all incoming edges to the intersecting metabolite originate.
        :type inNetwork: nx.DiGraph

        :param startingNode: Currently considered intersecting metabolite. Also starting point for the enumeration of elementary circuits from which to start the enumeration.
        :type startingNode: str

        :param nodeDeletelist: A list of nodes that are deleted from the network and substituted by two nodes, one that has only in- and outgoing edges in the in-network and one that has only in- and outgoing edges in the out-network.
        :type nodeDeletelist: list
        
        Returns:
            - newNetwork: The generated orientied network.
        '''
    global globalSubN 
    newNetwork = deepcopy(globalSubN)
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
    '''processCircuitsAll
    
        Upon invocation this function processes the enumreated elementary circuits either in parallel or not. It calls analyzeElementaryCircuits() to determine their CS equivalence class (MR-edges), compute the corresponding CS matrix, determine if it is Metzler or not and determine its spectral properties.

        Parameters
        ----------

        1. Global

        :param parallelBool: Boolean value indicating whether to process the circuits in parallel or not.
        :type parallelBool: bool

        :param equivClassLengthDict: Dictionary mapping the length of the CS equivalence class (number of MR-edges) to the number of identified CS equivalence classes with this length. Key: int (length of CS equivalence class), value: int (number of identified CS equivalence classes with this length).
        :type equivClassLengthDict: dict

        :param cycleLengthDict: Dictionary mapping the length of the elementary circuits (number of vertices) to the number of identified elementary circuits with this length. Key: int (length of elementary circuit), value: int (number of identified elementary circuits with this length).
        :type cycleLengthDict: dict

        :param species: String representing the currently analyzed species, which is used for the description of the tqdm progress bar.
        :type species: str

        :param elementaryCircuits: List of identified elementary circuits, where each elementary circuit is represented as a list of vertices (metabolite, reaction, metabolite, reaction, ...).
        :type elementaryCircuits: list

        :fluffleBool: Boolean value indicating if also fluffle equivalence classes should be enumerated. In this case the elementary circuits are stored in the list elementary circuits. Otherwise not.
        :type fluffleBool: bool

        :param noThreads: Integer representing the number of threads (Mac) or processes (Linux) to use for parallel processing. Only used if parallelBool is True.
        :type noThreads: int

        2. Local

        :param circuits: Generator obtained from nx.simple_cycles().
        :type circuits: generator

        :param description: String that is used as description for the tqdm progress bar.
        :type description: str  

        Returns:
            - circuitCounter: Integer representing the number of identified new CS equivalence classes.        
    '''
    global parallelBool, equivClassLengthDict, cycleLengthDict, species, elementaryCircuits, fluffleBool, noThreads
    #global allCircuitsPath
    breakBool = False
    n=0
    futureSet = set()
    circuitCounter=0
    if parallelBool==True:
        if sys.platform.startswith("linux"):
            executor = concurrent.futures.ProcessPoolExecutor()
        if sys.platform == "darwin":
            executor = concurrent.futures.ThreadPoolExecutor()
        else:
            executor = concurrent.futures.ProcessPoolExecutor()
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
        executor.shutdown()
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
    ''' processCircuits

        Distributor function: Upon invocation this function calls subfunctions to process the elementary circuits for identifying all autocatalytic CS matrices with irreducible Metzler part (processCircuitsCores()) or only cores (processCircuitsCores()) depending on the value of the global variable coreBool. The function returns the number of identified new CS equivalence classes after processing the given elementary circuits.

        Parameters
        ----------

        1. Global

        :param coreBool: Boolean value indicating whether to process only cores or all elementary circuits.
        :type coreBool: bool

        2. Local

        :param circuits: Generator obtained from nx.simple_cycles() representing the elementary circuits to process.
        :type circuits: generator

        :param leaf: Boolean value indicating whether the currently analyzed vertex of the partition tree is a leaf or not. This is used for the description of the tqdm progress bar.
        :type leaf: bool

        :param left: Boolean value indicating whether the currently analyzed vertex of the partition tree is a left child or not. This is used for the description of the tqdm progress bar.
        :type left: bool

        :param circuitCounter: Integer representing the number of identified new CS equivalence classes before processing the given elementary circuits. This is updated by the function and returned after processing the given elementary circuits.
        :type circuitCounter: int

        Returns:
            - circuitCounter: Integer representing the number of identified new CS equivalence classes after processing the given elementary circuits.
    '''
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
    ''' readArguments
    
        Upon invocation this function reads the command line arguments and stores them in variables. The function returns the following variables: 
        - inputXMLFilePath: String representing the file path to the original xml-file of the metabolic network, which is used for the generation of the stoichiometric matrix S and the corresponding dictionaries mapping metabolite and reaction identifiers to integer indices and vice versa.
        - inputPickleFile: String representing the file path to the input pickle file, which contains all the necessary data from partitionNetwork.py
        - circuitBound: Integer representing the maximum number of MR-edges contained in a CS equivalence classes corresponding to an elementary circuit.
        - checkNonMetzler: Boolean value indicating whether to check also CS equivalence classes with non-Metzler matrices for autocatalysis or not. If False, only CS equivalence classes with Metzler matrices are checked for autocatalysis.
        - noThreads: Integer representing the number of threads (Mac) or processes (Linux) to use for parallel processing. Only used if parallelBool is True.
        - equviClassBound: Integer representing the maximum number of MR-edges contained in a largewr CS equivalence class. 
        - fluffleBool: Boolean value indicating whether to also enumerate fluffle equivalence classes or not. If True, also fluffle equivalence classes are enumerated.
        - coreBool: Boolean value indicating whether to only identify cores or all autocatalytic CS equivalence classes. If True, only cores are identified. If False, all CS equivalence classes with autocatalytic CS matrices with irreducible Metzler part are identified.
        - parallelBool: Boolean value indicating whether to parallel processing should be enforced for processing of elementary circutis and assembly of larger CS equivalence classes.
        - species: String representing the currently analyzed species, which is used for the description of the tqdm progress bar.
        - outputPath: String representing the file path to the output directory in which all collected data is ought to be stored.

        Returns:
            - inputXMLFilePath
            - inputPickleFile
            - circuitBound
            - checkNonMetzler
            - noThreads
            - equviClassBound
            - fluffleBool
            - coreBool
            - parallelBool
            - species
            - outputPath
            as described above.
        '''
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


def writeDataToListsAndDicts(parameters:dict, aM:list, nAM:list, nMAC:list, nMnAC:list):
    ''' writeDataToListsAndDicts
    Upon invocation this function writes the data obtained from the analysis of CS equivalence classes into the corresponding entries of the parameters dictionary
    
    Parameters:
        
        :param parameters: The dictionary containing the parameters.
        :type parameters: dict

        :param aM: List with CS equivalence classes and corresponding autocatalytic Metzler CS matrices (equivalently Hurwitz-unstable).
        :type aM: list

        :param nAM: List with CS equivalence classes and corresponding non-autocatalytic Metzler CS matrices.
        :type nAM: list

        :param nMAC: List with CS equivalence classes and corresponding non-Metzler autocatalytic CS matrices.
        :type nMAC: list

        :param nMnAC: List with CS equivalence classes and corresponding non-Metzler non-autocatalytic CS matrices.
        :type nMnAC: list
    
        Returns:
            - None, but the function updates the parameters dictionary by writing the data obtained from the analysis of CS equivalence classes into the corresponding entries of the parameters dictionary.
    '''
    
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
    ''' writeDictionaryFromStoichiometricMatrix
        Upon invocation this function writes the stoichiometric matrix S into a dictionary, where the keys are tuples of the form (i,j) representing the row and column indices of the corresponding entry in the stoichiometric matrix S and the values are the corresponding entries of the stoichiometric matrix S. The function returns the generated dictionary.
        
        Parameters
        ----------
        :param S: Stoichiometric matrix S
        :type S: np.matrix

        Returns:
            - SDict: Dictionary where the keys are tuples of the form (i,j) representing the row and column indices of the corresponding entry in the stoichiometric matrix S and the values are the corresponding entries of the stoichiometric matrix S.
        '''
    SDict = {}
    for i in range(np.shape(S)[0]):
        for j in range(np.shape(S)[1]):
            SDict[(i,j)] = S[i,j]
    return SDict
#############################
#############################


def writeStoichiometricMatrixOutput(parameters:dict, path:str):
    ''' writeStoichiometricMatrixOutput
        Upon invocation this function writes the stoichiometric matrix S into a text file, where each line corresponds to a row of the stoichiometric matrix S and the entries are separated by spaces. The function does not return anything but writes the stoichiometric matrix S into a text file at the specified path. Was implemented for comparison with Gagrani et al., but is not used in the current version of the code.
        
        Parameters
        ----------
        
        :param parameters: The dictionary containing the parameters, which also contains the stoichiometric matrix S under the key "StoichiometricMatrix".
        :type parameters: dict

        :param path: String representing the file path to the output text file in which the stoichiometric matrix S is ought to be stored.
        :type path: str

        Returns:
            - None, but the function writes the stoichiometric matrix S into a text file at the specified path.
        '''

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
    ''' analyzeElementaryCircuitsCore
        Upon invocation this function determines the CS equivalence class of a given elementary circuit c, computes the corresponding CS matrix, determines if it is Metzler or not and determines its spectral properties. The function returns the following variables:
        
        Parameters
        ----------
        
        :param c: List of vertices representing an elementary circuit, where the vertices are ordered as follows: metabolite, reaction, metabolite, reaction.
        :type c: list
        
        Returns:
            - remove: Boolean value indicating whether the currently analyzed elementary circuit should be discarded from further analysis, i.e. if it contains a duplicate of an intersecting metabolite that was considered before and is now present in _in and _out version.
            - c: The input elementary circuit, which is returned for further processing if remove is False.
            - mrEdgeSet: Set of MR-edges contained in the given elementary circuit
            - unstable: Boolean value indicating whether the CS matrix corresponding to the given elementary circuit is Hurwitz-unstable or not. Only defined if remove is False.
            - metzler: Boolean value indicating whether the CS matrix corresponding to the given elementary circuit is Metzler or not. Only defined if remove is False.
    '''
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
    ''' assembleCores

        Upon invocation this function assembles larger CS equivalence classes, however, with the aim to identify only autocatalytic cores. It is the version for enumerating autocatalytic cores of assemebleLargerEquivalenceClasses(). Done in parallel or not depending on the users choice. 
        
        Parameters
        ----------

        :param parameters: The dictionary containing the parameters, which also contains the global variables and data structures necessary for the assembly of larger CS equivalence classes.
        :type parameters: dict

        :param Q: Deque containing the CS equivalence classes that are queued for assembly. 
        :type Q: deque

        :param E: Dictionary mapping CS equivalence classes to their properties, which also contains the global variables and data structures necessary for the assembly of larger CS equivalence classes. Key: frozenset of MR-edges representing a CS equivalence class, value: dictionary containing the properties of the corresponding CS equivalence class.
        :type E: dict

        :param speedCores: Set of candidates for autocatalytic cores that have been identified.
        :type speedCores: set

        Returns:
            - None, but the function updates the global variables and data structures necessary for the assembly of larger CS equivalence classes by assembling larger CS equivalence classes and identifying autocatalytic cores among them. 

        '''
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
    ''' callAssembleCythonCores

        Version of callAssembleCython() to detect autocatalytic cores. The only difference is that this function calls assembleCythonCores() instead of assembleCython() and that the global variables and data structures used in assembleCythonCores() are those corresponding to the assembly of cores, which are also used in the parallelized version of assembleCores().

        Parameters
    ----------

    1. Global 

    :param elemE: Storing CS equivalence classes for elementary circuits as keys with additional information on the equivalenc class in a dictionary as value. 
    :type elemE: dict

    :param M: Dictionary storing the CS equivlance classes containing a particular MR-edge. Key: One single MR-edge, value: Set of CS equivalence classes containing this MR-edge. 
    :type M: dict

    :param circuitIdMrEdgeDict: Dictionary storing the MR-edges corresponding to a certain elementary circuit. Key: int (elementary circuit identifier), value: frozenset of MR-edges of the key.
    :type circuitIdMrEdgeDict: dict

    :param bigS: Stoichiometric matrix S of the original metabolic network as a numpy array.
    :type bigS: np.array

    :param mID: Dictionary mapping metabolite identifiers to integer indices corresponding to the row indices of the stoichiometric matrix S. Key: metabolite identifier, value: integer index corresponding to the row index of the stoichiometric matrix S.
    :type mID: dict

    :param rID: Dictionary mapping reaction identifiers to integer indices corresponding to the column indices of the stoichiometric matrix S. Key: reaction identifier, value: integer index corresponding to the column index of the stoichiometric matrix S.
    :type rID: dict

    2. Local

    :param equivClass: Frozenset representing the currently analyzed equivalence class.
    :type equivClass: frozenset

    :param equivClassValues: Dictionary storing the information on the currently analyzed equivalence class. Key: "MR", "RM", "Predecessors", "Leaf", "Autocatalytic", "Metzler", "Update", "Visited", "Core", values: corresponding values
    :type equivClassValues: dict

    :param cutoff: Maximum length of CS equivalence classes that are assembled.
    :type cutoff: int

    Returns:
        - equivClass: The input equivClass, which is returned for further processing.
        - newEquivClasses: Dictionary storing the new equivalence classes that are assembled from the currently analyzed equivalence class and all overlapping CS equivalence classes. Keys: frozensets of MR-edges, value: dictionary with different information and datastructures. As an example: {"MR": mr, "RM": rm, "Predecessors": set(), "Leaf": True, "Autocatalytic": False, "Metzler": True, "Update": False, "Visited": False, "Core": False}. mr and rm are again dictionaries specifying the correspondence between metabolites and reactions for metabolite-to-reaction and reaction-to-metabolite edges, respectively.
        - change: Dictionary storing the changes in already existing equivalence classes for later use in assembleLargerEquivClassesParallel(). Keys: frozensets of MR-edges, value: Dictionary with different information and datastructures that are updated by the assembly of the currently analyzed equivalence class with all overlapping CS equivalence classes. 
    '''
    global elemE, M, circuitIdMrEdgeDict, bigS, mID, rID
    newEquivClasses, change = assembleCythonCores(equivClass, equivClassValues, elemE, M, circuitIdMrEdgeDict, cutoff, bigS, mID, rID)
    return equivClass, newEquivClasses, change
#############################
#############################


def checkEquivalenceClassCore(c, eqClass, autocatalytic):
    """ checkEquivalenceClassCore

    Version of checkEquivalenceClass for detecting autocartalytic cores. The only difference is that this function uses the global variables and data structures corresponding to the assembly of cores, which are also used in the parallelized version of assembleCores(). Upon invocation, this function determines information about the vertices 
    and edges of elementary circuits given on the list queue. Importantly, elementary circuits vertex and edge-sets (E and E1: only metabolite -> reaction edges) serve 
    as keys and elementary circuits are stored according to equivalence classes invoked by these properteis in different dictionaries.
    
    Parameters
    ----------
    
    1. Global
    
        :param E: Dictionary designated to store all CS equivalence classes. Keys: frozensets of MR-edges, value: dictionary with different information and datastructures. As an example: {"MR": mr, "RM": rm, "Predecessors": set(), "Leaf": True, "Autocatalytic": False, "Metzler": True, "Visited": False, "Core": False}. mr and rm are again dictionaries specifying the correspondence between metabolites and reactions for metabolite-to-reaction and reaction-to-metabolite edges, respectively.
        :type E : dict 

        :param Q: Contains all elementary circuits enumerated by the Johnsons-Algorithm from def(analysePartitionTree).
        :type Q: list
    
        :param M: Dictionary storing the CS equivlance classes containing a particular MR-edge. Key: One single MR-edge, value: Set of CS equivalence classes containing this MR-edge. 
        :type M: dict  

        :param circuitIdDict: Dictionary mapping cycle identifiers to elementary circuits. Key: int (cycle identifier), value: elementary circuit
        :type cycleIDDict: dict

        :param circuitIdMrEdgeDict: Dictionary mapping cycle identifiers to the set of MR-edges contained in the corresponding elementary circuit. Key: int (cycle identifier), value: frozenset of MR-edges contained in the corresponding elementary circuit
        :type circuitIdMrEdgeDict: dict

        :param speedCores: Set of candidates for autocatalytic cores that have been identified.
        :type speedCores: set

    2. Local

        :param c: List of vertices representing an elementary circuit.
        :type c: list

        :param eqClass: Set of MR-edges contained in the elementary circuit represented by c.
        :type eqClass: set

        :param autocatalytic: Boolean value indicating whether the CS matrix corresponding to the elementary circuit represented by c is autocatalytic or not.
        :type autocatalytic: bool

    Returns:
        - Boolean value indicating whether the equivalence class of the currently analyzed elementary circuit has already been added to the global variable E or not. If it has not been added, the function adds the equivalence class to E and updates the global variables M, circuitIdDict, circuitIdMrEdgeDict with the information on the currently analyzed elementary circuit and its equivalence class. If it has already been added, the function returns False and does not update any global variable.
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
        if sys.platform.startswith("linux"):
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=noThreads)
        elif sys.platform == "darwin":
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=noThreads)
        else:
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=noThreads)
        futureList = []
        while True:
            try:
                futureList.append(executor.submit(analyzeElementaryCircuitsCore, next(circuits)))
                n+=1
            except StopIteration as sti:
                breakBool = True
            if breakBool == True or n>10e6:
                for f in tqdm(concurrent.futures.as_completed(futureList), leave = False, total = n, desc= description+species):
                    try:
                        remove, circuit, eqClass, autocatalytic, metzler = f.result()
                        if remove == False:
                            l = len(circuit)
                            cycleLengthDict[l]=cycleLengthDict.setdefault(l,0)+1
                            if metzler == True:
                                if checkEquivalenceClassCore(circuit, eqClass, autocatalytic):
                                    circuitCounter+=1
                                    lequiv = len(eqClass)*2
                                    equivClassLengthDict[lequiv]=equivClassLengthDict.setdefault(lequiv,0)+1
                                    #cycleFile.write(str(circuit) + "\n")
                        del circuit, f
                    except Exception as exc:
                        print('%r generated an exception: %s', exc)
                futureList=[]
                n=0
                if breakBool==True:
                    break
    else:
        for c in circuits:
            remove, circuit, eqClass, autocatalytic, metzler = analyzeElementaryCircuitsCore(c)
            if remove == False:
                l = len(circuit)
                cycleLengthDict[l]=cycleLengthDict.setdefault(l,0)+1
                if metzler == True:
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

cycleLengthDict = {}
equivClassLengthDict = {}
E = {}                                                                            # \mathcal{E} im paper
elemE = {}
M = {}                                                                            # M in the draft, Dictionary Key: One! MR-edge only, Value: Set of cycles containing this MR-edge
circuitIdDict = {}                                                                # Dictionary Key: cycle-identifier (int), Value: Cycle
circuitIdMrEdgeDict = {}                                                          # Dictionary Key: Cycle-Identifier Value: Set of MR-edges contained in this cycle   
Q = deque()                                                                       # Q im paper
elementaryCircuits = []
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
analysePartitionTree(partitionTree, siblings, leaves, uRN, usefulNetwork, circuitBound)
parameters["cycleLengthDict"] = cycleLengthDict
totalTime = time.time()-timeStamp
parameters["TotalTime"] = totalTime
parameters["MaxRAM"]=maxRAM/(1024**3)

with open(outputPickleFilePath, "wb") as file:
    pickle.dump((parameters, partitionTree, siblings, leaves, uRN, usefulNetwork), file)

