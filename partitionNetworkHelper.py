# Partitioning Helper 

# Packages 
import networkx as nx
import libsbml
import sys
import sympy as sp
from sympy import * 
import concurrent.futures

def analyseOverlap(partitionTree:nx.DiGraph, network:nx.DiGraph, siblings:dict):
    maxOverlap = 0
    overlapLengthDict = {}
    overlapMetaboliteDict = {}
    for subG in partitionTree.nodes():
        subnetwork, metabolites, reactions = generateSubnetwork(subG, network)
        if len(partitionTree.in_edges(subG))==0:
            continue
        siblingSubG = siblings[subG]
        if len(partitionTree.in_edges(subG))!=0:
            siblingSubnetwork, siblingMetabolites, siblingReactions = generateSubnetwork(siblingSubG, network)
            overlapMetaboliteDict[(subG, siblingSubG)] = set(siblingMetabolites).intersection(set(metabolites))
            overlap = len(overlapMetaboliteDict[(subG, siblingSubG)])
            overlapLengthDict[overlap] = overlapLengthDict.setdefault(overlap, 0) + 1
            if overlap > maxOverlap:
                maxOverlap = overlap
    return maxOverlap, overlapLengthDict, overlapMetaboliteDict
#############################
#############################


def buildNetwork(model:libsbml.Model):
    # 0. Read parameters
    #model = parameters["model"]
    # 1 Define new variables
    metabolicNetwork = nx.DiGraph()
    # 1. Get list of reactions
    reactions = model.getListOfReactions()
    vertexIDs = {} 
    counter = 0
    for r in reactions:
        rName = r.getId()
        rFullName = r.getName()
        if not rFullName.startswith("R_"):
            rFullName = "R_" + rFullName
        if "BIOMASS" in rName:
            continue
        if "EX" in rName:
            continue
        if "transport" in rFullName.lower():
            continue
        reversible = r.getReversible()
        if reversible == True:
            rNameFW = rName + "_fw"
        else:
            rNameFW = rName
        educts = r.getListOfReactants()
        products = r.getListOfProducts()
        reactionFWID = counter
        counter+=1
        vertexIDs[rNameFW] = reactionFWID
        metabolicNetwork.add_node(reactionFWID)
        metabolicNetwork.nodes[reactionFWID]["Name"]=rNameFW
        metabolicNetwork.nodes[reactionFWID]["Type"]="Reaction"
        for e in educts:
            eSpecies = e.getSpecies()
            eductStoichiometry = e.getStoichiometry()
            if not eSpecies.startswith("M_"):
                eSpecies = "M_" + eSpecies
            if eSpecies not in vertexIDs:
                eSpeciesID = counter
                counter+=1
                vertexIDs[eSpecies] = eSpeciesID
                metabolicNetwork.add_node(eSpeciesID)
                metabolicNetwork.nodes[eSpeciesID]["Name"] = eSpecies
                metabolicNetwork.nodes[eSpeciesID]["Type"] = "Species"
            else:
                eSpeciesID = vertexIDs[eSpecies]
            metabolicNetwork.add_edge(eSpeciesID, reactionFWID)
            metabolicNetwork.edges[eSpeciesID, reactionFWID]["Stoichiometry"]=eductStoichiometry
        for p in products:
            pSpecies = p.getSpecies()
            productStoichiometry = p.getStoichiometry()
            if not pSpecies.startswith("M_"):
                pSpecies = "M_" + pSpecies
            if pSpecies not in vertexIDs:
                pSpeciesID = counter
                counter+=1
                vertexIDs[pSpecies] = pSpeciesID
                metabolicNetwork.add_node(pSpeciesID)
                metabolicNetwork.nodes[pSpeciesID]["Name"] = pSpecies
                metabolicNetwork.nodes[pSpeciesID]["Type"] = "Species"
            else:
                pSpeciesID = vertexIDs[pSpecies]
            metabolicNetwork.add_edge(reactionFWID, pSpeciesID)
            metabolicNetwork.edges[reactionFWID, pSpeciesID]["Stoichiometry"]=productStoichiometry
        if reversible == True:
            rNameRev = rName + "_rev"
            reactionRevID = counter
            counter +=1
            vertexIDs[rNameRev] = reactionRevID
            metabolicNetwork.add_node(reactionRevID)
            metabolicNetwork.nodes[reactionRevID]["Name"] = rNameRev
            metabolicNetwork.nodes[reactionRevID]["Type"] = "Reaction"
            for e in products:
                eSpecies = e.getSpecies()
                s = e.getStoichiometry()
                eSpeciesID = vertexIDs[eSpecies]
                metabolicNetwork.add_edge(eSpeciesID, reactionRevID)
                metabolicNetwork.edges[eSpeciesID, reactionRevID]["Stoichiometry"]=s
            for p in educts:
                pSpecies = p.getSpecies()
                s = p.getStoichiometry()
                pSpeciesID = vertexIDs[pSpecies]
                metabolicNetwork.add_edge(reactionRevID, pSpeciesID)
                metabolicNetwork.edges[reactionRevID, pSpeciesID]["Stoichiometry"]=s
    return metabolicNetwork, vertexIDs
#############################
#############################


def callCoefficientAnalyzer(A:sp.Matrix, noThreads:int):
    lamda = sp.Symbol("lamda")
    p = A.charpoly(lamda)
    coefficients = p.coeffs()
    DAG = nx.DiGraph()
    DAG.add_node("1")
    edgeDict = {}
    minusCoffDict = {}
    minusPlusCoffDict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=noThreads) as executor:
        futureSet = {executor.submit(computeEdges, coefficients[x], x, coefficients) for x in range(1, len(coefficients))}
        j =0
        for future in concurrent.futures.as_completed(futureSet):
            try:
                edgeDict[j], minusC, minusPlusC = future.result()
                minusCoffDict.update(minusC)
                minusPlusCoffDict.update(minusPlusC)
                j+=1
            except Exception as exc:
                print('%r generated an exception: %s' % (str(j), exc))
    print(minusPlusCoffDict)
#############################
#############################


def checkForMetzlerMatrix(S):
    for i in range(sp.shape(S)[0]):
        for j in range(sp.shape(S)[1]):
            if i == j:
                continue
            else:
                if S[i,j] < 0:
                    return False
    return True
#############################
#############################


def computeEdges(c, x, coefficients):
    edgeSet = set()
    summands = sp.Add.make_args(c)
    minusCoffs = {}
    minusPlusCoffs = {}
    for y in range(len(summands)):
        s = summands[y]
        if s.could_extract_minus_sign() == False:
            continue
        sNums = {atom for atom in s.atoms() if atom.is_number}
        if len(sNums)>1:
            sys.exit("Numbers bigger than zero, something is wrong here")
        if len(sNums) == 1:
            k = sNums.pop()
            symS = s.coeff(k)
        else:    
            symS = s
        edgeSet.add(("1", symS))
        minusCoffs[symS] = []
        minusPlusCoffs[symS] = []
        #  DAG.add_edge("1", symS)
        for z in range(x+1, len(coefficients)):
            largerCoefficients = coefficients[z]
            lCoeffs = largerCoefficients.coeff(symS)
            if lCoeffs == 0:
                continue
            else:
                lCoeffsList = sp.Add.make_args(lCoeffs)
            for w in range(len(lCoeffsList)):
                n = lCoeffsList[w]
                nNums = {atom for atom in n.atoms() if atom.is_number}
                if len(nNums)>1:
                    sys.exit("Numbers bigger than zero, something is wrong here")
                if len(nNums) == 1:
                    l = nNums.pop()
                    symN = n.coeff(l)
                else:
                    symN = n
                #DAG.add_edge(symS, symS*symN)
                edgeSet.add((symS, symS*symN))
                if n.could_extract_minus_sign()==True:
                    minusCoffs[symS] = minusCoffs.setdefault(symS, []) + [-symS*symN]
                else:
                    minusPlusCoffs[symS] = minusPlusCoffs.setdefault(symS, []) + [symS*symN]
    return edgeSet, minusCoffs, minusPlusCoffs
#############################
#############################


def createReactionNetwork(sCC:nx.DiGraph, reactions:set, inhibitors:dict):
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


def getAbundantMolecules(smallMoleculesSet:set, metabolicNetwork:nx.DiGraph, exclude:set):
    abundantMolecules = set()
    unneccessaryMolecules =set()
    for n in metabolicNetwork.nodes():
        node = metabolicNetwork.nodes[n]["Name"]
        if node.startswith("M_"):
            if node.endswith("_e"):
                unneccessaryMolecules.add(node)
                abundantMolecules.add(node)
            shortendNode = "_".join(node.split("_")[:-1])+"_"
            if shortendNode in exclude:
                abundantMolecules.add(node)
            if shortendNode in smallMoleculesSet:
                unneccessaryMolecules.add(node)
                abundantMolecules.add(node)
    return unneccessaryMolecules, abundantMolecules
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

