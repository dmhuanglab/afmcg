# afmcg
=============

:author: Huong TL Nguyen

Anisotropic Force-Matching Coarse-Graining of atomistic dynamical simulation models. 

Please contact me with any questions (<huong.nguyen@adelaide.edu.au>) or submit an issue!

Installation
------------

To get ``afmcg``, you can install it with pip::

    $ pip install afmcg

Algorithm
---------

This code contains three main processes: (1) transforming atomistic coordinates to anisotropic CG coordinates for a atomistic trajectory created using molecular dynamics simulation package LAMMPS, (2) building a matrix whose rows contain basis functions that can be linealy combined to estimate the atomistic forces on each CG particles. The basis functions are calculated from pairwise relative position and orientation between the central CG particle with its neighbouring particles within a cutoff distance, assuming additive pairwise interactions, and (3) reducing the matrix to a triangular size and solving the matrix using a least-squares optimization routine.

The code is designed so that calculation of the matrix can be run on several computer processing units (CPUs) using python multiprocessing package.

The current algorithm allows parametrization for systems with one CG particle type with only non-bonded interaction between the CG particles, and the CG particles having a uniaxial symmetry.

Examples
--------

The example in ``examples/main.py`` shows an example of python script to calculate the CG pairwise interaction between benzene molecules with each molecule being coarse-grained into one CG particle. The atomistic LAMMPS trajectory used for the example is ``examples/benzene500-300K-1atm.lammpstrj`` that contains 48 atomistic configuration frames from an equilibrium state of a simulation of 500 benzene molecules at temperature 300K and pressure 1atm. The output file that contains the coefficients of the basis functions to estimate the CG interaction using all 48 simulation frames is ``benzene500-300K-1atm.lammpstrj_coeff_frame_1_48_1``

A minimum example of a python script to parametrize the CG pairwise interaction between benzene molecules using the above input file is

.. code-block:: python
    from functools import partial
    import numpy as np
    from multiprocessing import Pool, freeze_support
    import os

    from matrix import construct_matrix
    from helper_functions import frame_split
    from solver_SEQHT import SEQHT, LS_solver

    # Required information of atomistic simulation model
    path = "./"
    file_name= "npt-cutoff-10.lammpstrj_small"
    num_atom_per_molecule= 12
    num_molecule= 500
    
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
    # Divide frame_list into blocks to build matrix and solve sequentially
    frame_block_lists = frame_split(frame_list,frame_block_size)

    def main(frames):
        pool = Pool(num_process)
        result = pool.map(partial(construct_matrix, path=path,file_name=file_name,r_min=r_min,r_max=r_max,num_r_center=num_r_center,k_max=k_max,num_molecule=num_molecule,num_atom_per_molecule=num_atom_per_molecule), frames)
        return result

    if __name__=="__main__":
        freeze_support()
        # Create /scratch directory:
        os.makedirs(os.path.dirname(path + "scratch/"), exist_ok=True)
        # Build matrix and perform sequential accumulation
        W = np.empty([0,1])
        for frames in frame_block_lists:
            matrix = main(frames)
            W = SEQHT(path,file_name,frame_list,frames,matrix,W)
        # Final solution written in a file in path directory:
        resid = LS_solver(path,file_name,frame_list,W)
        print("Force root-mean-squared error is %.4f (force unit)" % ((resid**2/(num_molecule*3*len(frame_list)))**(1/2)))

Descriptions and notes for the different required input variables are available in the documentation included in ``doc/README.md``

The example in ``examples/CG-potential.py`` shows how to plot the CG pair potential using the output basis coefficients.

Contributors and Acknowledgements
---------------------------------

I developed this code as a PhD student in the research group of Associate Professor David Huang (<https://huang-lab.org//>) at the University of Adelaide.

License
-------

This project is licensed under the CC-BY 4.0 license.
