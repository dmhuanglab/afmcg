"""
This module defines the a function that calculate the matrix block for 1 atomistic frame (3 rows of the whole matrix)
"""

import numpy as np
from math import sqrt,acos

from .pbf import grid_centre, radius_norm_1D, periodic_basis_force_torque_cubic_splines,grid_centre_nonuniform
from .read_lammps_data import read_atomistic_data, output_CG_data, box_boundaries_Lammps_file
from .coords_functions import compute_unwrapped_coords, global_to_body_transform, global_to_body_frame,right_handed_coord_system, min_image
from .orientation_functions import rotationMatrixToEulerAngles
from .helper_functions import clamp

def construct_matrix(frame,path,file_name,r_min,r_max,num_r_center,k_max,num_molecule,num_atom_per_molecule,spheroid):
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
    force_AA, torque_AA = output_CG_data(path,file_name,frame,num_atom_per_molecule,num_molecule,spheroid)
    matrix_f, matrix_t = [], []

    # Create files to save the matrix into hard disk
    matrix_file_f = open(path  + "scratch/" + file_name + "_matrix_force_" + str(frame),'w')
    matrix_file_f.close()
    matrix_file_t = open(path  + "scratch/" + file_name + "_matrix_torque_" + str(frame),'w')
    matrix_file_t.close()

    # Simulation box
    xboxlo,xboxhi,yboxlo,yboxhi,zboxlo,zboxhi = box_boundaries_Lammps_file(path,file_name,num_atom_per_molecule,num_molecule,frame)
    xbox = xboxhi - xboxlo
    xbox2 = 0.5*xbox
    ybox = yboxhi - yboxlo
    ybox2 = 0.5*ybox
    zbox = zboxhi - zboxlo
    zbox2 = 0.5*zbox
    
    # Calculate matrix components
    with open (path + "scratch/" + file_name + "_frame_" + str(frame) + "_CG_data",'r') as fin, open(path  + "scratch/" + file_name + "_matrix_force_" + str(frame),'a') as fout, open(path  + "scratch/" + file_name + "_matrix_torque_" + str(frame),'a') as tout:
        CG_coord = fin.readlines()
        for i in range(num_molecule):
            pbf_all_f,pbf_all_t = [], []
            matrix_xyz_f, matrix_xyz_t = [[] for _ in range(3)],[[] for _ in range(3)]
            
            ref_site_pos = np.array([float(CG_coord[i].split()[1]),float(CG_coord[i].split()[2]),float(CG_coord[i].split()[3])])
            ref_site_orient_x = np.reshape([float(CG_coord[i].split()[4]),float(CG_coord[i].split()[5]),float(CG_coord[i].split()[6])],(3,1))
            ref_site_orient_y = np.reshape([float(CG_coord[i].split()[7]),float(CG_coord[i].split()[8]),float(CG_coord[i].split()[9])],(3,1))
            ref_site_orient_z = np.reshape([float(CG_coord[i].split()[10]),float(CG_coord[i].split()[11]),float(CG_coord[i].split()[12])],(3,1))
            ref_site_orient_x, ref_site_orient_y, ref_site_orient_z=right_handed_coord_system(ref_site_orient_x, ref_site_orient_y, ref_site_orient_z)
            ref_site_orient_matrix = np.concatenate((np.transpose(ref_site_orient_x),np.transpose(ref_site_orient_y),np.transpose(ref_site_orient_z)),axis=0)
            neighbour_list = list(range(num_molecule))
            neighbour_list.remove(i)
            for j in neighbour_list:
                site_pos = np.array([float(CG_coord[j].split()[1]),
                        float(CG_coord[j].split()[2]),float(CG_coord[j].split()[3])])
                dx= min_image(ref_site_pos[0]-site_pos[0],xbox,xbox2)
                dy= min_image(ref_site_pos[1]-site_pos[1],xbox,xbox2)
                dz= min_image(ref_site_pos[2]-site_pos[2],xbox,xbox2)
                site_site_dist = sqrt(dx**2 + dy**2 + dz**2)
                if site_site_dist<0.1 or site_site_dist > cutoff:
                        pass
                else:
                    site_orient_x = np.reshape([float(CG_coord[j].split()[4]),float(CG_coord[j].split()[5]),float(CG_coord[j].split()[6])],(3,1))
                    site_orient_y = np.reshape([float(CG_coord[j].split()[7]),float(CG_coord[j].split()[8]),float(CG_coord[j].split()[9])],(3,1))
                    site_orient_z = np.reshape([float(CG_coord[j].split()[10]),float(CG_coord[j].split()[11]),float(CG_coord[j].split()[12])],(3,1))
                    site_orient_x, site_orient_y, site_orient_z=right_handed_coord_system(site_orient_x, site_orient_y, site_orient_z)
                    site_orient_matrix_glob = np.concatenate((np.transpose(ref_site_orient_x),np.transpose(ref_site_orient_y),np.transpose(ref_site_orient_z)),axis=0)
                    site_orient_matrix_body = global_to_body_transform(ref_site_orient_matrix,site_orient_x,site_orient_y,site_orient_z)
                    beta = rotationMatrixToEulerAngles(site_orient_matrix_body)[1]
                    # recalculate site_pos due to periodic boundaries
                    site_pos = np.array([ref_site_pos[0]- dx,ref_site_pos[1]- dy,ref_site_pos[2]- dz])
                    site_site_dir = (ref_site_pos - site_pos)/np.linalg.norm(ref_site_pos - site_pos)
                    A = acos(clamp(float(np.dot(site_site_dir,ref_site_orient_z)),-1,1))
                    B = acos(clamp(float(np.dot(site_site_dir,site_orient_z)),-1,1))
                    dcosA_dR = (ref_site_orient_z.flatten() - site_site_dir)*float(np.dot(site_site_dir,ref_site_orient_z))/site_site_dist
                    dcosB_dR = (site_orient_z.flatten() - site_site_dir)*float(np.dot(site_site_dir,site_orient_z))/site_site_dist
                    dcosbeta_di3 = site_orient_z.flatten() 
                    dcosA_di3 = site_site_dir
                    i3_unit = ref_site_orient_z.flatten() 
                    basis_var = [site_site_dist,beta,A,B]

                    pbf_f, pbf_t =  periodic_basis_force_torque_cubic_splines(center_r,k_max,basis_var,site_site_dir,dcosA_dR,dcosB_dR,dcosbeta_di3,dcosA_di3,i3_unit)
                    pbf_all_f.append(pbf_f)
                    pbf_all_t.append(pbf_t)
            #Sum for all neighbour sites
            pbf_all_f=[sum(i) for i in zip(*pbf_all_f)]
            pbf_all_t=[sum(i) for i in zip(*pbf_all_t)]
            for pbf_i in range(len(pbf_all_f)):
                for k in range(3):
                    matrix_xyz_f[k].append(pbf_all_f[pbf_i][k])
                    matrix_xyz_t[k].append(pbf_all_t[pbf_i][k])
            matrix_f.extend(matrix_xyz_f)
            matrix_t.extend(matrix_xyz_t)

        matrix_f = np.column_stack([np.array(matrix_f),np.array(force_AA)])
        matrix_t = np.column_stack([np.array(matrix_t),np.array(torque_AA)])
        np.savetxt(fout, matrix_f, fmt="%.6e")
        np.savetxt(tout, matrix_t, fmt="%.6e")
    return 
