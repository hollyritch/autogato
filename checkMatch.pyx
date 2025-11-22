# check_match.pyx
from libc.stdlib cimport malloc, free
from cython.parallel import prange
from itertools import chain
import numpy as np
cimport numpy as cnp
import scipy as sc
# cython: language_level=3
# Tell NumPy to use the modern API:
cdef extern from *:
    """
    #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
    """

def assembleCython(frozenset equivClass, dict equivClassValues, dict elemE, dict M, dict circuitIdMrEdgeDict, int cutoff):
    """
    Assemble new equivalence classes from a given class.
    """

    cdef list intersecEquivClasses
    cdef dict changeE, newEquivClasses
    cdef frozenset newFrozen
    cdef object interEqCl, interEquivClassValues, newMR
    cdef bint plausible

    # Compute intersections
    intersecEquivClasses, changeE = getIntersectingEquivClassesParallelCython(equivClass, M, circuitIdMrEdgeDict)
    newEquivClasses = {}
    for interEqCl in intersecEquivClasses:
        newEquivClass = equivClass | interEqCl
        if len(newEquivClass) <= cutoff:
            newFrozen = frozenset(newEquivClass)
            interEquivClassValues = elemE[interEqCl]
            plausible, newMR = checkMatch(equivClassValues, interEquivClassValues)
            if plausible:
                newEquivClasses[newFrozen] = {
                    "MR": newMR,
                    "RM": 0,
                    "Predecessors": {equivClass, interEqCl},
                    "Leaf": False,
                    "Autocatalytic": False,
                    "Metzler": True,
                    "Update": False,
                    "Visited": False,
                    "Core": False
                }

    return newEquivClasses, changeE
#############################
#############################


def checkMatch(dict equivClassDict, dict interEquivClassDict):
    cdef dict[int, int] newMR = {}

    cdef dict[int, int] mr1 = equivClassDict["MR"]

    cdef dict[int, int] mr2 = interEquivClassDict["MR"]
    cdef dict[int, int] rm2 = interEquivClassDict["RM"]

    cdef int m1, m2, r1, r2

    for m1 in mr1:
        r1 = mr1[m1]
        r2 = mr2.get(m1, -1)
        if r2 != -1:
            if r1 == r2:
                newMR[m1] = r1
            else:
                return False, None
        else:
            m2 = rm2.get(r1,-1)
            if m2 !=-1:
                return False, None
            else:
                newMR[m1] = r1

    for m2 in mr2:
        if m2 not in mr1:
            newMR[m2] = mr2[m2]

    return True, newMR
#############################
#############################


def getIntersectingEquivClassesParallelCython(frozenset equivClass, dict M, dict circuitIdMrEdgeDict):
    """
    Determine equivalence classes intersecting with the current class.
    """
    cdef list intersecEquivClasses = []
    cdef dict changeE = {}
    cdef set all_circuits = set(chain.from_iterable(M[e] for e in equivClass))
    cdef object c, cEqCl, value

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




# ------------------------------------------------------------------------#
#                               Cores                                     #
# ------------------------------------------------------------------------#

def assembleCythonCores(frozenset equivClass, dict equivClassValues, dict elemE, dict M, dict circuitIdMrEdgeDict, int cutoff, S, dict mID, dict rID):
    """
    Assemble new equivalence classes from a given class.
    """

    cdef list intersecEquivClasses
    cdef dict changeE, newEquivClasses
    cdef frozenset newFrozen
    cdef object interEqCl, interEquivClassValues, newMR
    cdef bint plausible
    cdef tuple result

    # Compute intersections
    intersecEquivClasses, changeE = getIntersectingEquivClassesParallelCythonCores(equivClass, equivClassValues, M, circuitIdMrEdgeDict, elemE)
    newEquivClasses = {}
    for interEqCl in intersecEquivClasses:
        newEquivClass = equivClass | interEqCl
        if len(newEquivClass) <= cutoff:
            newFrozen = frozenset(newEquivClass)
            interEquivClassValues = elemE[interEqCl]
            plausible, newMR = checkMatch(equivClassValues, interEquivClassValues)
            if plausible:
                subS, metzler = computeSubstochasticMatrixForSetOfMREdgesCython(S, mID, rID, newEquivClass)
                if metzler == True:
                    autocatalytic = determineStabilityCython(subS)
                    newEquivClasses[newFrozen] = {
                        "MR": newMR,
                        "RM": 0,
                        "Predecessors": {equivClass, interEqCl},
                        "Leaf": False,
                        "Autocatalytic": autocatalytic,
                        "Metzler": metzler,
                        "Update": False,
                        "Visited": False,
                        "Core": autocatalytic
                    }
    return newEquivClasses, changeE
#############################
#############################


def computeSubstochasticMatrixForSetOfMREdgesCython(S, dict mID, dict rID, frozenset newEquivClass):
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
    cdef bint metzler = True
    cdef int k = len(newEquivClass)
    cdef cnp.ndarray subS = np.zeros((k, k), dtype=float)
    cdef dict mRDict = {}

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
    return subS, metzler
#############################
#############################


def determineStabilityCython(T:np.matrix):
    cdef int k = np.shape(T)[0]
    cdef bint unstable = False
    cdef object sM
    cdef object lamda 

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


def getIntersectingEquivClassesParallelCythonCores(frozenset equivClass, dict equivClassValues, dict M, dict circuitIdMrEdgeDict, dict elemE):
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
    
    cdef list intersecEquivClasses = []
    cdef dict changeE = {}
    cdef set all_circuits = set(chain.from_iterable(M[e] for e in equivClass))
    cdef int c
    cdef frozenset cEqCl
    cdef dict value 
    autocatalytic=equivClassValues["Autocatalytic"]
    for c in all_circuits:            
        cEqCl = circuitIdMrEdgeDict[c]
        if equivClass == cEqCl:
            continue
        elif equivClass < cEqCl:
            value = elemE.setdefault(cEqCl, {})
            value.setdefault("Predecessors", set()).add(equivClass)
            value["Leaf"] = False
            if autocatalytic==True:
                value["Core"]=False
        elif cEqCl < equivClass:
            equivClassValues.setdefault("Predecessors", set()).add(cEqCl)
            equivClassValues["Leaf"] = False
            if elemE[cEqCl]["Autocatalytic"]==True:
                equivClassValues["Core"]=False
        else:
            if autocatalytic == True:
                continue
            elif elemE[cEqCl]["Autocatalytic"]==True:
                continue
            else:
                intersecEquivClasses.append(cEqCl)    
    return intersecEquivClasses, changeE
#############################
#############################

