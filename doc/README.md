## PREPARING ATOMISTIC TRAJECTORY
------------

When running and outputting LAMMPS trajectory files to be used with the afmcg package, the following minimum atomistic details need to be included in the trajectory:

mol = molecule ID

x, y, z = unscaled atom coordinates

mass = atom mass

fx, fy, fz = forces on atoms

Also see LAMMPS manual on dump command for more details at <https://docs.lammps.org/dump.html>

## REQUIRED INPUT INFORMATION
------------

### Below are examples of minimum information that must be placed in the input file:

path = "path-to-LAMMPS-file/" # directory to the LAMMPS atomistic trajectory file, note the syntax of directory may vary for different OS

file_name = "example.lammpstrj" # name of LAMMPS atomitic trajectory file

num_atom_per_molecule = 12 # number of atoms in each CG particle/molecule

num_molecule = 500 # number of CG particles/molecules

r_min = 3 # minimum separation distance between CG particles, in the same distance unit with LAMMPS atomistic trajectory

r_max = 10 # maximum separation distacne between CG paritcles (cutoff of CG interaction), in the same distance unit with LAMMPS atomistic trajectory

num_r_center = 23 # number of basis functions of inter-particle distance

k_max = 8 # maximum order or periodic basis functions of relative orientation variables

frame_start = 1 # atomistic frame to start CG parametrization

frame_end = 48 # atomistic frame to end CG parametrization (equal or smaller to the total number of frames in the atomistic trajectory file)

frame_skip = 1 # step between frames used for parametrization, if set to 1 all frames from frame_start to frame_end is used

frame_block_size = 12 # number of frames per block to perform sequential triangularization 

num_process = 12 # number of CPUs to run on (ideally equal or smaller than frame_block_size so all CPUs can be run at once)

## Note about deciding the number of frames per block to perform sequential triangularization
------------

The purpose of deviding the total frames into blocks with smaller size is to reduce the memory requirement to solve the matrix. For example, with the above setting for the basis functions, there will be 23x5x5x5 = 2875 basis functions in total, so the matrix will have 2875 columns. With the above atomistic settings, each block contains number of rows for 12 frames, each frame contains the basis functions for 500 CG particles, so the total number of rows for each block is 12x500x3 = 18000 (multiply by 3 because there are 3 force components for each CG particle). Performing a Householder Triangularization for a 18000x2875 is a lot cheaper than a matrix that contains 4 times as many rows (which is the case if the block size is 48). Users are to check the appropriate block size to fit the memory available on their machine before running the afmcg code.
