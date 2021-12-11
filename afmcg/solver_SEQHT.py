import os
import shutil
import numpy as np

def SEQHT(path,file_name,frame_list,frames,matrix,W):
    """
    Sequential householder transformation to reduce a matrix to a square matrix
    which has the same least-squares solution and residual.
    """    
    matrix = np.vstack(matrix) 
    with open(path + "scratch/matrix_temporary","w") as fout:
        np.savetxt(fout, matrix, fmt="%.5e")
    matrix = np.loadtxt(path + "scratch/matrix_temporary")
    if frames[0] == frame_list[0]:
        W = matrix
    else:
        W = np.concatenate((W,matrix),axis=0)
    W = np.linalg.qr(W)[1]
    #Remove files in path/scratch directory:
    for i in frames:
        os.remove(path + "scratch/" + file_name + "_frame_" + str(i))
        os.remove(path + "scratch/" + file_name + "_frame_" + str(i) + "_matrix")
        os.remove(path + "scratch/" + file_name + "_frame_" + str(i) + "_CG_data")
    return W

def LS_solver(path,file_name,frame_list,W):
    """
    Solve a matrix using linear least-squares method and write out solution file
    Return the residual of the least-squares solution
    """
    x = np.linalg.lstsq(W[:,0:-1],W[:,-1],rcond=None)[0]
    frame_skip = frame_list[1] - frame_list[0]
    with open(path + file_name + "_coeff_frame_{}_{}_{}".format(frame_list[0],frame_list[-1],frame_skip),"w") as outfile:
        for j in x:
            outfile.write(str(float(j)) + "\t")
            outfile.write("\n")
    resid = abs(W[-1][-1])
    #Remove path/scratch directory:
    shutil.rmtree(path + "scratch")
    return (resid)
