"""
This module defines the a function that calculate the matrix block for 1 atomistic frame (3 rows of the whole matrix)
"""

import numpy as np

from .pbf import grid_centre, radius_norm_1D, periodic_basis_force_derivative_cubic_splines,grid_centre_nonuniform
from .read_lammps_data import calculate_data_points_uniaxial, pbc_interaction_image, read_atomistic_data, output_CG_data
from .helper_functions import countLineWithPattern

def construct_matrix(frame,path,file_name,r_min,r_max,num_r_center,k_max,num_molecule,num_atom_per_molecule):
    """
    Contruct force matrix for a frame (a system configuration) from molecular 
    dynamic trajectory.
    The matrix should have (3*num_molecule) rows corresponding to the number of 
    atomistic force data points, and (num_r_center*k_max**3+1) columns
    corresponding to the number of basis functions+1. 
    The last column is the atomistic force on a CG particle.
    """
    cutoff = r_max
    center_r = grid_centre_nonuniform(r_min,r_max,num_r_center)

    # Create a file to save the matrix into hard disk
    read_atomistic_data(path,file_name,num_atom_per_molecule,num_molecule,frame)
    force_AA = output_CG_data(path,file_name,frame,num_atom_per_molecule,num_molecule)
    pbc_interaction_image(path,file_name,num_atom_per_molecule,num_molecule,frame,cutoff)
    calculate_data_points_uniaxial(path,file_name,frame,cutoff,num_molecule)
    matrix = []
    # Calculate matrix components
    with open(path + "scratch/" + file_name + "_frame_" + str(frame) + "_matrix",'r') as fin:
        data = fin.readlines()
        num_neigh = countLineWithPattern(path + "scratch/" + file_name +
                          "_frame_" + str(frame) + "_matrix", num_molecule)
        for i in range(0,num_molecule):
            matrix_xyz = [[] for _ in range(3)]
            data_i = data[(sum(num_neigh[0:i])*int(i>0)):
                              (sum(num_neigh[0:i])*int(i>0)+num_neigh[i])]
            data_i = [x.split("\t") for x in data_i]
            data_mat = [[float(x) for x  in sublist[1:5]] for sublist in data_i]
            data_mat = np.array([np.array(x) for x in data_mat])
            R_unit = [np.array([float(x) for x  in sublist[5].split(",")]) for sublist in data_i]
            dcosA_dR = [np.array([float(x) for x  in sublist[6].split(",")]) for sublist in data_i]
            dcosB_dR = [np.array([float(x) for x  in sublist[7].split(",")]) for sublist in data_i]
            
            pbf_dR_all =[]
            for j in range(len(data_mat)):
                pbf_dR = periodic_basis_force_derivative_cubic_splines(center_r,k_max,data_mat[j],R_unit[j],dcosA_dR[j],dcosB_dR[j])
                pbf_dR_all.append(pbf_dR)
            pbf_dR_all=[sum(i) for i in zip(*pbf_dR_all)]

            for pbf_i in range(len(pbf_dR)):
                for k in range(3): # for force matching in each direction x, y, z
                    matrix_xyz[k].append(pbf_dR_all[pbf_i][k])
            matrix.extend(matrix_xyz)
        matrix = np.column_stack([np.array(matrix),np.array(force_AA)])
    return matrix
