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
    '''
        assembleCython

        Upon invocation this function assembles new equivalence classes from a given class by finding intersecting equivalence classes and checking whether the MR relationship is consistent with the new class.

        Parameters
        ----------
            :param equivClass: Set of MR-edges of the current CS equivalence class to check
            :type equivClass: frozenset

            :param equivClassValues: Dictionary containing the values of the current equivalence class, e.g. MR relationship, predecessors, etc.
            :type equivClassValues: dict

            :param elemE: Dictionary containing all equivalence classes corresponding to elementary circuits with their values, e.g. MR relationship, predecessors, etc.
            :type elemE: dict

            :param M: Dictionary mapping one MR-edge to the set of elementary circuits containing this particular MR-edge
            :type M: dict

            :param circuitIdMrEdgeDict: Dictionary mapping an elementary circuit identifier to the set of MR-edges contained in this circuit
            :type circuitIdMrEdgeDict: dict

            :param cutoff: Maximum size of CS- equivalence classes to be assembled in terms of MR-edges
            :type cutoff: int
        
        Returns
        -------
            :return newEquivClasses: Dictionary of new equivalence classes assembled from the current class
            :rtype newEquivClasses: dict    

            :return changeE: Dictionary of equivalence classes whose values have been changed due to the assembly of new equivalence classes
            :rtype changeE: dict                   
    '''

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
    ''' checkMatch
        
        Upon invocation this function checks whether the MR relationship of two equivalence classes is consistent with the MR relationship of the new class that would be assembled from these two classes. If the MR relationship is consistent, then the MR relationship of the new class is returned. In particular, there should be exactly one perfect matching between metabolites and reactions in the new class.

    Parameters
    ----------

        :param equivClassDict: Dictionary containing the currently considered CS equivalence class as key and their values, predecessors, etc.
        :type equivClassDict: dict

        :param interEquivClassDict: Dictionary containing the currently considered intersecting CS equivalence classes as keys and their values, predecessors, etc.
        :type interEquivClassDict: dict

    Returns
    -------
        :return plausible: Boolean value specifying whether the MR relationship of the new class is consistent with the MR relationship of the two classes that are being assembled
        :rtype plausible: bool

        :return newMR: Dictionary specifying the MR relationship of the new class, if the MR relationship is consistent, otherwise None
        :rtype newMR: dict or None        
        '''
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
    '''getIntersectingEquivClassesParallelCython

        Upon invocation this function determines those equivalence classes (sets of MR-edges) that intersect with the current cycle of interest.

    Parameters
    ----------  

        :param equivClass: Set of MR-edges of the current cycle to check.
        :type equivClass: frozenset

        :param M: Dictionary mapping each MR-edge to the set of circuits containing it.
        :type M: dict

        :param circuitIdMrEdgeDict: Dictionary mapping each circuit identifier to the set of MR-edges contained in this circuit.
        :type circuitIdMrEdgeDict: dict

    Returns
    -------

        :return intersecEquivClasses: List of equivalence classes intersecting with the current class.
        :rtype intersecEquivClasses: list

        :return changeE: Dictionary of equivalence classes whose values have been changed due to the assembly of new equivalence classes
        :rtype changeE: dict
    '''
    
    
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
    '''assembleCythonCores

    Upon invocation this function assembles new equivalence classes from a given class by finding intersecting equivalence classes and checking whether the MR relationship is consistent with the new class. This function is the version designated for autocatalytic cores only instead of finding all CS equivalence classes with corresponding autocatalytic CS matrix with irreducible Metzler part.

    Parameters
    ----------

        :param equivClass: Set of MR-edges of the current CS equivalence class to check.
        :type equivClass: frozenset 

        :param equivClassValues: Dictionary containing the values of the current equivalence class, e.g. MR relationship, predecessors, etc.
        :type equivClassValues: dict

        :param elemE: Dictionary containing all equivalence classes corresponding to elementary circuits with their values, e.g. MR relationship, predecessors, etc.
        :type elemE: dict

        :param M: Dictionary mapping one MR-edge to the set of elementary circuits containing this particular MR-edge
        :type M: dict

        :param circuitIdMrEdgeDict: Dictionary mapping an elementary circuit identifier to the set of MR-edges contained in this circuit
        :type circuitIdMrEdgeDict: dict

        :param cutoff: Maximum size of CS- equivalence classes to be assembled in terms of MR-edges
        :type cutoff: int

        :param S: Stochastic matrix of the network
        :type S: np.matrix

        :param mID: Dictionary mapping each metabolite to its corresponding row index in the stochastic matrix S
        :type mID: dict 

        :param rID: Dictionary mapping each reaction to its corresponding column index in the stochastic matrix S
        :type rID: dict

        Returns
        -------

        :return newEquivClasses: Dictionary of new equivalence classes assembled from the current class
        :rtype newEquivClasses: dict    

        :return changeE: Dictionary of equivalence classes whose values have been changed due to the assembly of new equivalence classes
        :rtype changeE: dict
    '''

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
        if elemE[interEqCl]["Autocatalytic"]:
            continue
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


def determineAutocatalycityLP(S:np.matrix):
    '''determineAutocatalycityNonMetzler
    
        Upon invocation this function checks if a given CS matrix that is not a Metzler matrix is autocatalytic by checking if it has a positive real eigenvalue. Since the matrix is not a Metzler matrix, we cannot use the Perron-Frobenius theorem to check for the existence of a positive real eigenvalue, which is why we have to use linear programming to check if there exists a positive vector v such that Sv>0, which is equivalent to the existence of a positive real eigenvalue. The function returns a boolean value indicating whether the given CS matrix that is not a Metzler matrix is autocatalytic or not.
        
        Parameters
        ----------
        :param S: k x k CS matrix that is not a Metzler matrix, where k is the number of MR-edges in the given set of MR-edges representing the CS equivalence class for which the autocatalycity is to be checked. The columns of S are re-ordered according to the given MR relationship, which defines a perfect matching thus a CS. Accordingly, for an edge (m,r) then r represents the i-th column if and only if m represents the i-th row.
        :type S: np.matrix  
        
        Returns:
            - autocatalytic: Boolean value indicating whether the given CS matrix that is not a Metzler matrix is autocatalytic or not.'''
    cdef int k = np.shape(S)[0]
    cdef cnp.ndarray A = (-1)*S
    cdef cnp.ndarray b = np.zeros((k, 1), dtype=float)
    cdef cnp.ndarray c = np.zeros((1, k), dtype=float)
    cdef list bounds = []

    print("Determining autocatalyticity via LP")
    b = np.zeros((k,1))    
    c = np.ones((1,k))
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


def determineStabilityCython(T:np.matrix):
    '''determineStabilityCython

        This function determines the stability of a Metzler CS matrix by checking the real parts of its eigenvalues. 
        This funciton is invoked only if only autocatalytic cores are intented to be found.
        
        Parameters
        ----------
        :param T: The k x k CS matrix for which stability is to be checked. k is the number of MR-edges.
        :type T: np.matrix
        
        Returns:
            - unstable: Boolean value indicating whether the circuit is unstable.
    '''
    cdef int k = np.shape(T)[0]
    cdef bint unstable = False
    cdef object sM
    cdef object lamda 
    
    unstable = determineAutocatalycityLP(T)

    if k>=3:
        try:
            sM = sc.sparse.csr_matrix(T)
            lamda = sc.sparse.linalg.eigs(sM, k=1, which = "LR", return_eigenvectors = False)
            if round(np.real(lamda[0]), 5)>0:
                unstable = True   
        except:
            try:
                for lamda in np.linalg.eigvals(T):
                    if round(np.real(lamda),5)>0:        
                        unstable = True
                        break
            except:
                unstable = determineAutocatalycityLP(T)       
    else:
        try:
            for lamda in np.linalg.eigvals(T):
                if round(np.real(lamda),5)>0:        
                    unstable = True
                    break
        except:
            unstable = determineAutocatalycityLP(T)
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

