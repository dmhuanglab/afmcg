import numpy as np
import os
import h5py

from afmcg.solver_SEQHT import SEQHT, LS_solver

# Required information about atomistic simulations
path = "./"
file_name= "benzene500-300K-1atm.lammpstrj"
num_molecule= 500

# Required information about parametrization
r_min = 3
r_max = 10
scale = 1.63 # 1/scale torque matrix, normally chosen as the molecular average semi-axis length
num_r_center = 4
k_max = 2
frame_start = 1
frame_end = 8
frame_skip = 1 # should choose a factor of (frame_end - frame_start)
num_frames = (frame_end - frame_start)/frame_skip + 1
block_size = 8 # block_size is # of frames used for sequential accumulation
file_ext = str(frame_start) + "_" + str(frame_end) + "_" + str(frame_skip)
n_basis = int((num_r_center)*(k_max/2+1)*(k_max/2+1)*(k_max/2+1))

#### Read in the force and torque matrices and perform sequential accumulation to produce a reduced square matrix. Then the coefficient is solved by a least square solver of this matrix. 
# sequential accumulation for force matrix
W = np.empty([0,n_basis+1])
W = SEQHT(path,file_name,"force_" + file_ext,num_frames,block_size,num_molecule,W)
# sequential accumulation for torque matrix
W = SEQHT(path,file_name,"torque_" + file_ext,num_frames,block_size,num_molecule,W)
# Solution and rmse
resid = LS_solver(path,file_name,file_ext,W)
with open(path + file_name + "_rmse_force_torque_scale_"+ str(scale) +"_frame_" +file_ext,"w") as fout:
    fout.write("Root-mean-squared error is %.4f (force unit)" % ((resid**2/(num_molecule*3*2*num_frames))**(1/2)))
# GZIP file after SEQHT, W can be used in the next sequential accumulation, for example if you decide to add more frames for force and torque matching 
with h5py.File(path + file_name + "_matrix_force_torque_frame_"+ file_ext + "_SEQHT.h5","w") as hf:
    W_matrix = hf.create_dataset('W_matrix', (n_basis+1,n_basis+1), compression='gzip', compression_opts=9, chunks=True)
    W_matrix[:]=W.reshape((-1, n_basis+1))     





