"""
This module defines general-purpose functions.
"""

from collections import Counter, OrderedDict
import numpy as np

def clamp(n, minn, maxn):
    "Restrict number n in a range [minn,maxn]"
    return max(min(maxn, n), minn)

class OrderedCounter(Counter, OrderedDict):
    'Counter that remembers the order elements are first encountered'

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)

def countLineWithPattern(file, num_molecule):
    mol_ID = [("site"+str(i)+"_") for i in np.linspace(1,num_molecule,num_molecule,dtype=int)] 
    results = OrderedCounter(mol_ID)
    myfile = open(file)
    for line in myfile:
        for pattern in mol_ID:
            if pattern in line:
                results[pattern] +=1
    # For some reasons counter returns one extra count
    results = np.array([(i[1]-1) for i in list(results.items())])
    return results

def frame_split(a, chunk):
    n = -(-len(a)//chunk)
    k, m = divmod(len(a), n)
    return list(a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def get_coeff(path,coeff_file):
    """
    Get coefficients from coefficient file
    Return: (m,1) matrix where m is the number of basis function coefficients
    """
    coeff = []
    with open(path + coeff_file,"r") as infile:
        for line in infile:
            data = [float(i) for i in line.rstrip('\t\n').split("\t")]
            coeff.append(data)
    coeff = np.matrix(coeff)
    return coeff
