# afmcg
=============

:author: Huong TL Nguyen

Anisotropic Force-Matching Coarse-Graining of atomistic dynamical simulation models. 

Please contact me with any questions (<huong.nguyen@adelaide.edu.au>) or submit an issue!

Installation
------------

To get ``afmcg``, you can install it with pip::

    $ pip install git+https://github.com/dmhuanglab/afmcg@v1.0.1

Alternatively, you can download the source code and run your python script from the mother directory 

Algorithm
---------

This code contains three main processes: (1) transforming atomistic coordinates to anisotropic CG coordinates for a atomistic trajectory created using molecular dynamics simulation package LAMMPS, (2) building a matrix whose rows contain basis functions that can be linealy combined to estimate the atomistic forces and torques on each CG particles. The basis functions are calculated from pairwise relative position and orientation between the central CG particle with its neighbouring particles within a cutoff distance, assuming additive pairwise interactions, and (3) reducing the matrix to a triangular size and solving the matrix using a least-squares optimization routine.

The code is designed so that calculation of the matrix can be run on several computer processing units (CPUs) using python multiprocessing package.

The current algorithm allows parametrization for systems with one CG particle type with only non-bonded interaction between the CG particles, and the CG particles having a uniaxial symmetry.

Examples
--------

The example in ``examples/main-build-matrix.py`` and ``examples/main-solve-matrix.py`` shows an example of python scripts to calculate the CG pairwise interaction between benzene molecules with each molecule being coarse-grained into one CG particle. The ``examples/main-build-matrix.py`` calculate the force and torque matrices, while the ``examples/main-solve-matrix.py`` solve the matrix equation using a sequential accumlation method. The atomistic LAMMPS trajectory used for the example is ``examples/benzene500-300K-1atm.lammpstrj`` that contains 48 atomistic configuration frames from an equilibrium state of a simulation of 500 benzene molecules at temperature 300K and pressure 1atm. The output file that contains the coefficients of the basis functions to estimate the CG interaction using all 8 simulation frames is ``benzene500-300K-1atm.lammpstrj_coeff_frame_1_8_1``, and other output files contains force and torque matrices, and the reduced square matrix after sequential accumulation that can be useful for analysis. 

Descriptions and notes for the different required input variables are available in the documentation included in ``doc/README.md``

The example in ``examples/CG-potential.py`` shows how to plot the CG pair potential using the output basis coefficients.

Contributors and Acknowledgements
---------------------------------

I developed this code as a PhD student in the research group of Associate Professor David Huang (<https://huang-lab.org//>) at the University of Adelaide.

License
-------

This project is licensed under the CC-BY 4.0 license.
