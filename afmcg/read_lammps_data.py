"""
This module defines functions to read and process LAMMPS dump trajectory files.
"""

from itertools import islice
import numpy as np
from math import acos

from .coords_functions import compute_unwrapped_coords

def column_name_Lammps_file(path,file_name):
    """
    Ouput columns of interested information such as: coordinates x,y,z, mass, forces
    fx, fy, fz, charge.
    """
    with open(path + file_name, "r") as input:
        for line in islice(input,8,9):
            col_var = line.split()
            mol_id = [i for i,x in enumerate(col_var) if x== "mol"][0] - 2
            x_col = [i for i,x in enumerate(col_var) if x== "x"][0] - 2
            y_col = [i for i,x in enumerate(col_var) if x== "y"][0] - 2
            z_col = [i for i,x in enumerate(col_var) if x== "z"][0] - 2
            fx_col = [i for i,x in enumerate(col_var) if x== "fx"][0] - 2
            fy_col = [i for i,x in enumerate(col_var) if x== "fy"][0] - 2
            fz_col = [i for i,x in enumerate(col_var) if x== "fz"][0] - 2
            mass_col = [i for i,x in enumerate(col_var) if x== "mass"][0] - 2
    return (mol_id,x_col,y_col,z_col,fx_col,fy_col,fz_col,mass_col)

def read_atomistic_data(path,file_name,num_atom_per_molecule,num_molecule,frame):
    """
    Read LAMMPS dump file and output coordinates and forces information

    Parameters:
    x_col,y_col,z_col,mass_col,fx_col,fy_col,fz_col = column containing
    x,y,z,mass,fx,fy,fz in LAMMPS dump file
    frame = which frame to be read

    Returns:
    A file containing x_col,y_col,z_col,fx_col,fy_col,fz_col,mass_col
    """
    # Lines to read from LAMMPS dump file
    first_line = (frame-1)*(num_atom_per_molecule*num_molecule+9)+9
    last_line = frame*(num_atom_per_molecule*num_molecule+9)-1

    # Write out files of interested time frames:
    mol_id,x_col,y_col,z_col,fx_col,fy_col,fz_col,mass_col = column_name_Lammps_file(path,file_name)
    with open(path + file_name, "r") as input, open(path + "scratch/" + file_name + "_frame_" + str(frame),"w") as output:
        for line in islice(input, first_line, last_line+1):
            data = line.split()[mol_id],line.split()[x_col],line.split()[y_col],line.split()[z_col],line.split()[fx_col],\
            line.split()[fy_col],line.split()[fz_col],line.split()[mass_col]
            output.write("\t".join(data) + "\n")
    with open (path + "scratch/" + file_name + "_frame_" + str(frame),"r+") as input:
        data = input.readlines()
        data.sort(key=lambda l: float(l.split("\t")[0]),reverse=False)
        input.seek(0)
        input.truncate()
        input.write("".join(data))
    return

def box_boundaries_Lammps_file(path,file_name,num_atom_per_molecule,num_molecule,frame):
    """
    Output boundaries of simulation box set for a simulation with dump file
    "path/file_name"
    """
    boundary_x_line = (frame-1)*(num_atom_per_molecule*num_molecule+9)+5

    with open(path + file_name, "r") as lammps_file:
        for line in islice(lammps_file, boundary_x_line, boundary_x_line+1):
            xboxlo = float(line.split(" ")[0])
            xboxhi = float(line.split(" ")[1].rstrip('\n'))
        for line in islice(lammps_file, 0, 1):
            yboxlo = float(line.split(" ")[0])
            yboxhi = float(line.split(" ")[1].rstrip('\n'))
        for line in islice(lammps_file, 0, 1):
            zboxlo = float(line.split(" ")[0])
            zboxhi = float(line.split(" ")[1].rstrip('\n'))
    return (xboxlo,xboxhi,yboxlo,yboxhi,zboxlo,zboxhi)

def output_CG_data(path,file_name,frame,num_atom_per_molecule,num_molecule,spheroid):
    """
    Output CG data out for each time frame.
    """
    xyz=[]
    m=[]
    site_ID = 1
    # Box boundaries:
    xboxlo,xboxhi,yboxlo,yboxhi,zboxlo,zboxhi = box_boundaries_Lammps_file(path,file_name,num_atom_per_molecule,num_molecule,frame)
    # Read file:
    force_AA = []
    torque_AA = []
    with open (path + "scratch/" + file_name + "_frame_" + str(frame),'r') as input, open(path + "scratch/" + file_name + "_frame_" + str(frame) + "_CG_data","w") as output:
        while True:
            try:
                data = [next(input) for _ in range(num_atom_per_molecule)]
                x = [float(i.split()[1]) for i in data]
                y = [float(i.split()[2]) for i in data]
                z = [float(i.split()[3]) for i in data]
                fx = [float(i.split()[4]) for i in data]
                fy = [float(i.split()[5]) for i in data]
                fz = [float(i.split()[6]) for i in data]
                m = [float(i.split()[7]) for i in data]

                # Unwrapping atom coords under periodic boundary conditions:
                x,y,z = compute_unwrapped_coords(x,y,z,xboxlo,xboxhi,yboxlo,yboxhi,zboxlo,zboxhi)
                # Calculating the centre of mass
                center = np.array([np.dot(x,m)/sum(m),np.dot(y,m)/sum(m),np.dot(z,m)/sum(m)])
                # Calculating vectors connecting center of mass and each atoms:
                x = np.reshape(x, (num_atom_per_molecule,1))
                y = np.reshape(y, (num_atom_per_molecule,1))
                z = np.reshape(z, (num_atom_per_molecule,1))
                m_matrix = np.diag(m)
                xyz = np.concatenate((x,y,z), axis=1)
                coord = np.matrix(xyz) - center

                # Calculating moment of inertia:
                weighted_coord = m_matrix*coord
                inertia_dot = np.array(np.dot(coord.transpose(), weighted_coord))
                inertia = -inertia_dot
                inertia[0][0] = inertia_dot[1][1] + inertia_dot[2][2]
                inertia[1][1] = inertia_dot[0][0] + inertia_dot[2][2]
                inertia[2][2] = inertia_dot[0][0] + inertia_dot[1][1]

                e_values, e_vectors = np.linalg.eig(inertia)

                order = np.argsort(e_values)
                eval1, eval2, eval3 = e_values[order]
                if spheroid == "oblate":
                    z_prime = np.squeeze(np.asarray(e_vectors[:,order[2]]))
                    y_prime = np.squeeze(np.asarray(e_vectors[:,order[1]]))
                    x_prime = np.squeeze(np.asarray(e_vectors[:,order[0]]))
                elif spheroid == "prolate":
                    z_prime = np.squeeze(np.asarray(e_vectors[:,order[0]]))
                    y_prime = np.squeeze(np.asarray(e_vectors[:,order[1]]))
                    x_prime = np.squeeze(np.asarray(e_vectors[:,order[2]]))
                output.write("{} {} {} {} {} {} {} {} {} {} {} {} {} \n".format(
                    site_ID,center[0],center[1],center[2],x_prime[0],x_prime[1],
                    x_prime[2],y_prime[0],y_prime[1],y_prime[2],z_prime[0],
                    z_prime[1],z_prime[2]))

                force_AA.extend([sum(fx),sum(fy),sum(fz)])
                torque = np.sum([np.cross(coord[i],np.array([fx[i],fy[i],fz[i]])) for i in range(len(fx))],axis=0).tolist()[0]
                torque_AA.extend([torque[0],torque[1],torque[2]])
                site_ID +=1
                
            except StopIteration:
                break
    return(force_AA,torque_AA)
