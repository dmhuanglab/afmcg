import os
import shutil
import numpy as np
import h5py

def read_dense_matrix_in_block_from_h5_file(path,file_name,file_ext,block_index,block_size_full):
    """
    Load the LQ matrix from a matrix data file in h5 compressed format.
    """
    with h5py.File(path + file_name + "_matrix_" + file_ext + "_gzip.h5",'r') as hf:
        data_all = hf.get('matrix')
        data = data_all[block_size_full*(block_index-1):block_size_full*block_index,:]
    return data

def SEQHT(path,file_name,file_ext,num_frames,block_size,num_molecule,W):
    """
    Sequential householder transformation to reduce a matrix to a square matrix
    which has the same least-squares solution and residual.
    return: 
    W: squared matrix
    resid: force-torque residue 
    """    
    block_size_full = int(3*num_molecule*block_size)
    q = int(3*num_molecule*num_frames/block_size)
    for t in range(1,q+1):
        At = read_dense_matrix_in_block_from_h5_file(path,file_name,file_ext,t,block_size_full)
        W = np.concatenate((W,At),axis=0)
        # Series of Householder transformation to make W become triangular form, similar to QR decomposition
        W = np.linalg.qr(W)[1]
    return (W)

def LS_solver(path,file_name,file_ext,W):
    """
    Solve a matrix using linear least-squares method and write out solution file
    Return the residual of the least-squares solution
    """
    x = np.linalg.lstsq(W[:,0:-1],W[:,-1],rcond=None)[0]
    with open(path + file_name + "_coeff_frame_"+file_ext,"w") as outfile:
        for j in x:
            outfile.write(str(float(j)) + "\t")
            outfile.write("\n")
    resid = abs(W[-1][-1])
    #Remove path/scratch directory:
    #shutil.rmtree(path + "scratch")
    return (resid)
