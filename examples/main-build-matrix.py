from functools import partial
import numpy as np
import os
import h5py

from multiprocessing import Pool, freeze_support
#from afmcg.matrix import construct_matrix
#from afmcg.helper_functions import frame_split
#from afmcg.solver_SEQHT import SEQHT, LS_solver

from matrix import construct_matrix
from helper_functions import frame_split

# Required information about atomistic simulations
path = "./"
file_name= "benzene500-300K-1atm.lammpstrj"
num_atom_per_molecule= 12
num_molecule= 500
spheroid = "oblate" # Choose spheroid type, "oblate" "prolate"

# Required information about parametrization
r_min = 3
r_max = 10
num_r_center = 4 #24
k_max = 2 #8
frame_start = 1
frame_end = 8
frame_skip = 1
frame_list = np.ndarray.tolist(np.arange(frame_start,frame_end+1,frame_skip, dtype = int))
frame_block_size = 8
num_process = 8

frame_block_lists = frame_split(frame_list,frame_block_size)

# Define a function for parametrization in parallel
def main(frames):
    pool = Pool(num_process)
    result = pool.map(partial(construct_matrix, path=path,file_name=file_name,r_min=r_min,r_max=r_max,num_r_center=num_r_center,k_max=k_max,num_molecule=num_molecule,num_atom_per_molecule=num_atom_per_molecule,spheroid=spheroid), frames)
    return result

if __name__=="__main__":
    freeze_support()
    #Create /scratch directory:
    os.makedirs(os.path.dirname(path + "scratch/"), exist_ok=True)
    #h5 file to store the force and torque matrices separately:
    num_line = ((frame_end-frame_start)/frame_skip+1)*3*num_molecule
    line_length = (num_r_center)*(k_max/2+1)*(k_max/2+1)*(k_max/2+1) + 1
    with h5py.File(path + file_name + "_matrix_force_" + str(frame_start) + "_" + str(frame_end) + "_" + str(frame_skip) + "_gzip.h5",'w') as hf:
        hf.create_dataset('matrix', (num_line,line_length), compression='gzip', compression_opts=9, chunks=True)
    with h5py.File(path + file_name + "_matrix_torque_" + str(frame_start) + "_" + str(frame_end) + "_" + str(frame_skip) + "_gzip.h5",'w') as hf:
        hf.create_dataset('matrix', (num_line,line_length), compression='gzip', compression_opts=9, chunks=True)
    # Build force and torque matrices
    for frames in frame_block_lists:
        main(frames)
        for i in frames:
            with h5py.File(path + file_name + "_matrix_force_" + str(frame_start) + "_" + str(frame_end) + "_" + str(frame_skip) + "_gzip.h5",'a') as hf_force, h5py.File(path + file_name + "_matrix_torque_" + str(frame_start) + "_" + str(frame_end) + "_" + str(frame_skip) + "_gzip.h5",'a') as hf_torque:
                force_matrix, torque_matrix = hf_force['matrix'], hf_torque['matrix']
                index_i = frame_list.index(i)
                data_force = np.loadtxt(path + "scratch/" + file_name + "_matrix_force_" + str(i))
                data_torque = np.loadtxt(path + "scratch/" + file_name + "_matrix_torque_" + str(i))
                force_matrix[index_i*3*num_molecule:(index_i+1)*3*num_molecule] = data_force
                torque_matrix[index_i*3*num_molecule:(index_i+1)*3*num_molecule] = data_torque
                #os.remove(path + "scratch/" + file_name + "_matrix_force_" + str(i))
                #os.remove(path + "scratch/" + file_name + "_matrix_torque_" + str(i))
                os.remove(path + "scratch/" + file_name + "_frame_" + str(i))
                os.remove(path + "scratch/" + file_name + "_frame_" + str(i) + "_CG_data")
