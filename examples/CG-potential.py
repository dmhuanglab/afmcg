import numpy as np
from matplotlib import pyplot as plt

from pbf import pbf_matrix
from helper_functions import get_coeff

r_min=3
r_max=10
num_r_center = 23
k_max = 8

path= "./"
coeff_file = "benzene500-300K-1atm.lammpstrj_coeff_frame_1_48_1"

# Get coefficients
coeff = get_coeff(path,coeff_file)

# Compute CG pair potential for face-to-face configuration
# beta, phi, theta in radians
x_face_to_face = np.linspace(3,r_max,200)
beta = 0 # angle between symmetry axis of uniaxial particle
phi = 0 # angle between inter-site vector and symmetry axis of particle 1
theta = 0 # angle between inter-site vector and symmetry axis of particle 2

data = np.array(np.meshgrid(x_face_to_face_10000_U,beta,phi,theta)).T.reshape(-1,4).tolist()
matrix_array = pbf_matrix(r_min,r_max,num_r_center,k_max,data)
U_face_to_face = np.asarray(matrix_array*coeff).flatten()

# Set potential to 0 at cutoff
U_face_to_face = U_face_to_face_10000 - U_face_to_face_10000[-1] 

# Plot CG pair potential versus inter-particle distance for face-to-face configuration
plt.plot(x_face_to_face_10000_U,U_face_to_face_10000)
plt.ylim(-2,5)
plt.xlim(2,10)
