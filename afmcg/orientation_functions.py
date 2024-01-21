"""
This module defines functions that calculate various orientational quantities.
"""

import numpy as np
from math import cos,sin,atan2,sqrt,radians
               
def right_handed_coord_system(x,y,z):
    """
    Making a set of 3 vectors (x,y,z) a right-handed coordinate system
    Parameters:
    x, y, z: column array vectors representing the principal axes
    """
    det_matrix = np.linalg.det(np.concatenate((x,y,z),axis=1))
    if det_matrix < 0:
        x = -x
    else:
        x = x
    return(x,y,z)

def quat_to_mat(quat):
    """
    Return rotation matrix from quaternion
    quat is an array of 4 elements
    """
    w2 = quat[0]*quat[0]
    i2 = quat[1]*quat[1]
    j2 = quat[2]*quat[2]
    k2 = quat[3]*quat[3]
    twoij = 2.0*quat[1]*quat[2]
    twoik = 2.0*quat[1]*quat[3]
    twojk = 2.0*quat[2]*quat[3]
    twoiw = 2.0*quat[1]*quat[0]
    twojw = 2.0*quat[2]*quat[0]
    twokw = 2.0*quat[3]*quat[0]

    m1 = []
    m1.extend([w2+i2-j2-k2])
    m1.extend([twoij-twokw])
    m1.extend([twojw+twoik])
    m1 = np.asarray(m1)

    m2 = []
    m2.extend([twoij+twokw])
    m2.extend([w2-i2+j2-k2])
    m2.extend([twojk-twoiw])
    m2 = np.asarray(m2)
    
    m3 = []
    m3.extend([twoik-twojw])
    m3.extend([twojk+twoiw])
    m3.extend([w2-i2-j2+k2])
    m3 = np.asarray(m3)

    return (m1,m2,m3)

def compute_quaternion_from_principal_axes(x,y,z):
    """
    Parameters:
    x, y, z: column array vectors representing the principal axes
    
    Returns: quaternions
    quat_w: scalar part of the quaternion
    quat_i: quaternion along the i axis of the quaternion vector part
    quat_j: quaternion along the j axis of the quaternion vector part
    quat_k: quaternion along the k axis of the quaternion vector part  
    
    Modified from: https://github.com/lammps/lammps
    math_extra.cpp
    Copyright (2003) Sandia Corporation.
    """
    # Right-handed coordinate system:
    x,y,z = right_handed_coord_system(x,y,z)

    # Squares of quaternion components:
    q0sq = 0.25 * (x[0] + y[1] + z[2] + 1.0)
    q1sq = q0sq - 0.5 * (y[1] + z[2])
    q2sq = q0sq - 0.5 * (x[0] + z[2])
    q3sq = q0sq - 0.5 * (x[0] + y[1])
    # Some component must be greater than 1/4 since they sum to 1
    # compute other compomemts from it
    q = np.empty(4, dtype=float) 
    if q0sq >= 0.25:
        q[0] = sqrt(q0sq)
        q[1] = (y[2] - z[1]) / (4.0*q[0])
        q[2] = (z[0] - x[2]) / (4.0*q[0])
        q[3] = (x[1] - y[0]) / (4.0*q[0])
    elif q1sq >= 0.25:
        q[1] = sqrt(q1sq)
        q[0] = (y[2] - z[1]) / (4.0*q[1])
        q[2] = (y[0] + x[1]) / (4.0*q[1])
        q[3] = (x[2] + z[0]) / (4.0*q[1])
    elif q2sq >= 0.25:
        q[2] = sqrt(q2sq)
        q[0] = (z[0] - x[2]) / (4.0*q[2])
        q[1] = (y[0] + x[1]) / (4.0*q[2])
        q[3] = (z[1] + y[2]) / (4.0*q[2])
    elif q3sq >= 0.25:
        q[3] = sqrt(q3sq)
        q[0] = (x[1] - y[0]) / (4.0*q[3])
        q[1] = (z[0] + x[2]) / (4.0*q[3])
        q[2] = (z[1] + y[2]) / (4.0*q[3])
    norm = 1/sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
    q[0] = norm*q[0]
    q[1] = norm*q[1]
    q[2] = norm*q[2]
    q[3] = norm*q[3]
    return (q) 

def compute_principal_axes_from_q(q):
    """
    Computing principal axes from quaternions
    ############################
    Parameters:
    q = [quat_w,quat_i,quat_j,quat_k]
    where
    quat_w: scalar part of the quaternion
    quat_i: quaternion along the i axis of the quaternion vector part
    quat_j: quaternion along the j axis of the quaternion vector part
    quat_k: quaternion along the k axis of the quaternion vector part

    #############################

    Returns: 
    x, y, z: column array vectors representing the principal axes
    
    Modified from: https://github.com/lammps/lammps
    math_extra.cpp
    Copyright (2003) Sandia Corporation.
    """
    x,y,z = np.empty(3, dtype=float)

    x[0] = q[0]*q[0] + q[1]*q[1] - q[2]*q[2] - q[3]*q[3]
    x[1] = 2.0 * (q[1]*q[2] + q[0]*q[3])
    x[2] = 2.0 * (q[1]*q[3] - q[0]*q[2])

    y[0] = 2.0 * (q[1]*q[2] - q[0]*q[3])
    y[1] = q[0]*q[0] - q[1]*q[1] + q[2]*q[2] - q[3]*q[3]
    y[2] = 2.0 * (q[2]*q[3] + q[0]*q[1])

    z[0] = 2.0 * (q[1]*q[3] + q[0]*q[2])
    z[1] = 2.0 * (q[2]*q[3] - q[0]*q[1])
    z[2] = q[0]*q[0] - q[1]*q[1] - q[2]*q[2] + q[3]*q[3]

    return(x,y,z)

def eulerAnglesToRotationMatrix(theta) :
    """
    Input:
    Theta = an array with 3 element corresponding to 3 Euler angles (degrees). 
    
    Return:
    R_z1,R_x2,Rz3 = arrays of vectors that define the rotation
    R = Rotation matrix
    """
    theta = [radians(i) for i in theta]
    R_z1 = np.array([[cos(theta[0]),    -sin(theta[0]),    0],
                    [sin(theta[0]),    cos(theta[0]),     0],
                    [0,                     0,                      1]
                    ])
                           
    R_x2 = np.array([[1,         0,                  0                   ],
                    [0,         cos(theta[1]), -sin(theta[1]) ],
                    [0,         sin(theta[1]), cos(theta[1])  ]
                    ])
                 
    R_z3 = np.array([[cos(theta[2]),    -sin(theta[2]),    0],
                    [sin(theta[2]),    cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                        
    R = np.dot(R_z1, np.dot( R_x2, R_z3 ))
     
    return(R_z1,R_x2,R_z3,R)

def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 
def rotationMatrixToEulerAngles(R) :
    """
    Returns Euler Angles from Rotation matrix, in radians
    Rotation Matrix: numpy matrix with columns as body-fixed principal axes
    """
    assert(isRotationMatrix(R))
     
    sy = sqrt(R[0,2] * R[0,2] +  R[1,2] * R[1,2])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = atan2(R[0,2] , -R[1,2])
        y = atan2(sy,R[2,2])
        z = atan2(R[2,0], R[2,1])
    else:
        x = atan2(R[1,0], R[0,0])/2
        y = atan2(sy,R[2,2])
        z = atan2(R[1,0], R[0,0])/2
    return [x, y, z]
