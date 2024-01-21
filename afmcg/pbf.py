"""
This module contains functions to calculate basis functions
"""

import numpy as np
from math import cos, sin, pi

def grid_centre(var_min,var_max,grid):
    """
    Compute centres for grids of data range in term of individual variables
    """
    num_centre = int((var_max-var_min)/grid)
    centre = np.linspace(var_min+grid/2,var_max-grid/2,num_centre)
    return(centre)

def grid_centre_nonuniform(var_min,var_max,N):
    """
    Compute centres for grids of data range in term of individual variables
    N is number of centers
    """
    centre = []
    for i in range(0,N):
        centre.append(var_max-(var_max-var_min)*cos(i*pi/(2*(N-1)))) 
    return(centre)

def pbf_meshgrid(center_r):
    """
    """
    k1 = np.arange(0,9,2)
    k2 = np.arange(0,9,2)
    k3 = np.arange(0,9,2)
    pbf_meshgrid = np.array(np.meshgrid(center_r,k1,k2,k3)).T.reshape(-1,4)
    return pbf_meshgrid
    
def radius_norm_1D(centre,x,grid):
    """
    Computing the norm of distance from a point vector(x)
    about a point vector(centre) in 1D
    """
    d_norm = ((x-centre)*2/grid)
    return(d_norm)

def cosine_der(k,x):
    """
    x is cos(y)
    evaluate derivative of cos(ky) wrt x=cos(y) for k = 0,1,2,3,4,...
    """
    if sin(x) == 0 or k==0:
        cos_der = 0
    else:
        cos_der = (k*sin(k*x))/sin(x)
    return cos_der
    
def nat_cubic_spline_basis(x,x_c,x_c_all):
    """
    Calculate natural cubic spline basis functions
    """
    ppart = lambda t: np.maximum(0, t)
    cube = lambda t: t*t*t
    numerator = (cube(ppart(x - x_c)) - cube(ppart(x - x_c_all[-1])))
    denominator = x_c_all[-1] - x_c
    return (numerator / denominator)

def nat_cubic_spline_basis_dR(x,x_c,x_c_all):
    """
    Calulate first derivative of natural cubic spline basis functions
    """
    ppart = lambda t: np.maximum(0, t)
    square = lambda t: t*t
    numerator = (3*square(ppart(x - x_c)) - 3*square(ppart(x - x_c_all[-1])))
    denominator = x_c_all[-1] - x_c
    return (numerator / denominator)

def periodic_basis_cubic_splines(center_r,k_max,data):
    """
    Evaluate periodic basis function (pbf) for CG intermolecular potential. Output is a set of basis function calculated for each neighbor of a CG site
    
    Parameters:
    pbf_i: 4D array containing important indices to construct basis function
    data: 4D array (r, beta, A, B)
    R_unit: 3D unit vector
    A_d_R: 3D vector
    B_d_R: 3D vector
    grid_r

    Returns:
    pbf: float value
    """

    pbf = []
    r,beta,A,B = data[:]

    # Periodic basis function terms
    for i in range(-2,len(center_r)-2):
        cub_spl_fixed = nat_cubic_spline_basis(r,center_r[-2],center_r)
        if i==-2:
            cub_spl = 1
        elif i==-1:
            cub_spl = r
        else:
            cub_spl = nat_cubic_spline_basis(r,center_r[i],center_r) - cub_spl_fixed
        for k1 in range(0,k_max+1,2):
            gk1 = cos(k1*beta)
            for k2 in np.arange(0,k_max+1,2):
                gk2 = cos(k2*A)
                for k3 in np.arange(0,k_max+1,2):
                    gk3 = cos(k3*B)
                    pbf.append(cub_spl*gk1*gk2*gk3)
    return pbf

def periodic_basis_force_torque_cubic_splines(center_r,k_max,basis_var,R_unit,dcosA_dR,dcosB_dR,dcosbeta_di3,dcosA_di3,i3_unit):
    """
    Evaluate periodic basis function (pbf) first erivative wrt CG site-site vectors (pbf_dR),
    for basis functions for intermolecular forces. Output is a set of basis function calculated
    for each neighbor of a CG site

    Parameters:
    pbf_i: 4D array containing important indices to construct basis function
    data: 4D array (r, beta, A, B)
    R_unit: 3D unit vector
    A_d_R: 3D vector
    B_d_R: 3D vector
    grid_r

    Returns:
    pbf_dR: 3D vector
    pbf_di3: 3D vector
    """
    r,beta,A,B = basis_var[:]
    pbf_dR,pbf_di3 = [], []

    # Periodic basis function terms
    for i in range(-2,len(center_r)-2):
        cub_spl_fixed = nat_cubic_spline_basis(r,center_r[-2],center_r)
        cub_spl_fixed_dR = nat_cubic_spline_basis_dR(r,center_r[-2],center_r)*R_unit
        if i==-2:
            cub_spl = 1
            cub_spl_dR = np.array([0,0,0])
        elif i==-1:
            cub_spl = r
            cub_spl_dR = R_unit
        else:
            cub_spl = nat_cubic_spline_basis(r,center_r[i],center_r) - cub_spl_fixed
            cub_spl_dR = nat_cubic_spline_basis_dR(r,center_r[i],center_r)*R_unit - cub_spl_fixed_dR
        for k1 in range(0,k_max+1,2):
            gk1 = cos(k1*beta)
            dgk1_dR = np.array([0,0,0])
            dgk1_di3 = cosine_der(k1,beta)*dcosbeta_di3
            for k2 in np.arange(0,k_max+1,2):
                gk2 = cos(k2*A)
                dgk2_dR = cosine_der(k2,A)*dcosA_dR
                dgk2_di3 = cosine_der(k2,A)*dcosA_di3
                for k3 in np.arange(0,k_max+1,2):
                    gk3 = cos(k3*B)
                    dgk3_dR = cosine_der(k3,B)*dcosB_dR
                    # The minus sign in front of each pbf_dR element is due to F= -dU/dR
                    pbf_dR.append(-(cub_spl_dR*gk1*gk2*gk3 + dgk1_dR*cub_spl*gk2*gk3
                            + dgk2_dR*cub_spl*gk1*gk3 + dgk3_dR*cub_spl*gk1*gk2))
                    pbf_di3.append(-np.cross(i3_unit,(dgk1_di3*cub_spl*gk2*gk3
                            + dgk2_di3*cub_spl*gk1*gk3)))
    return(pbf_dR,pbf_di3)

def pbf_matrix(r_min,r_max,num_r_center,k_max,data):
    """
    Compute pbf matrix for a data list that contains elementary lists such as
    [x,beta,phi,theta] for x,beta,phi,theta are the separation distance and
    angles that define relative orientations between two uniaxial particles
    
    center_r: list of knots for basis functions of inter-particle distance
    k_max: maximum order of periodic basis functions
    data: list of lists [[r_1,beta_1,phi_1,theta_1],...,[r_n,beta_n,phi_n,theta_n]]

    Return:
    matrix array with size (len(data),len(center_r)*(k_max/2+1)**3 
    """
    # Centre points for site-site distance variable:
    center_r = grid_centre_nonuniform(r_min,r_max,num_r_center)

    matrix_array = []
    matrix_append = matrix_array.append
    for i in data:
        pbf = periodic_basis_cubic_splines(center_r,k_max,i)
        matrix_append(pbf)
    matrix_array = np.matrix(matrix_array)
    return matrix_array
