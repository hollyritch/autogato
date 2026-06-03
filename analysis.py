import numpy as np
import scipy as sc
import sys
import pickle 

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
            if len(subS)>1:
                if i==j:
                    if subS[i][j]>0:
                        sys.exit("ERRRRRRORR, CS matrix is not a CS matrix")
    return subS, metzler
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
    k = np.shape(S)[0]
    A = (-1)*S
    b = np.zeros((k,1))    
    c = np.ones((1,k))
    bounds = []
    for i in range(k):
        bounds.append((1, None)) 
        b[i][0]=-1
    result = sc.optimize.linprog(c=c, A_ub=A, b_ub=b, bounds=bounds)
    if result["success"]==True:
        return True, result.x
    else:
        return False, np.zeros((np.shape(S)[1]))
#############################
############################# 

with open("/scratch/richard/Autocatalysis/cycleData/EColiCore/partitionTreeData0.pkl", "rb") as file:
    parameters, partitionTree, siblings, leaves, uRN, usefulNetwork = pickle.load(file)

### Most important is the parameters dictionary as it contains all data relevant for your analysis