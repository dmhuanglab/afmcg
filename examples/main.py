from functools import partial
import numpy as np
import os

from multiprocessing import Pool, freeze_support
from afmcg.matrix import construct_matrix
from afmcg.helper_functions import frame_split
from afmcg.solver_SEQHT import SEQHT, LS_solver


# Required information about atomistic simulations
path = "./"
file_name= "benzene500-300K-1atm.lammpstrj"
num_atom_per_molecule= 12
num_molecule= 500

# Required information about parametrization
r_min = 3
r_max = 10
num_r_center = 23
k_max = 8
frame_start = 1
frame_end = 48
frame_skip = 1
frame_list = np.ndarray.tolist(np.arange(frame_start,frame_end+1,frame_skip, dtype = int))
frame_block_size = 12
num_process = 12

frame_block_lists = frame_split(frame_list,frame_block_size)

# Define a function for parametrization in parallel
def main(frames):
    pool = Pool(num_process)
    result = pool.map(partial(construct_matrix, path=path,file_name=file_name,r_min=r_min,r_max=r_max,num_r_center=num_r_center,k_max=k_max,num_molecule=num_molecule,num_atom_per_molecule=num_atom_per_molecule), frames)
    return result

if __name__=="__main__":
    freeze_support()
    #Create /scratch directory:
    os.makedirs(os.path.dirname(path + "scratch/"), exist_ok=True)
    #Build matrix and perform sequential accumulation
    W = np.empty([0,1])
    for frames in frame_block_lists:
        matrix = main(frames)
        W = SEQHT(path,file_name,frame_list,frames,matrix,W)
    ##Final solution written in a file in path directory:
    resid = LS_solver(path,file_name,frame_list,W)
    print("Force root-mean-squared error is %.4f (force unit)" % ((resid**2/(num_molecule*3*len(frame_list)))**(1/2)))
