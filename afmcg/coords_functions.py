"""
This module defines functions that calculate molecular coordinates.
"""

import numpy as np
import sys

def min_image(dx,xbox,xbox2):
    """
    Calculates minimum image separations in one dimension x using 
    coordinates of lower and upper limits of the simulation box,
    xboxlo and xboxhi, and box width and half width, xbox and xbox2.
    
    Returns dx
    """
    while (dx > xbox2):
        dx = dx - xbox
    while (dx < -xbox2):
        dx = dx + xbox
    return(dx)

def compute_unwrapped_coords(x,y,z,xboxlo,xboxhi,yboxlo,yboxhi,zboxlo,zboxhi):
    """
    Takes lists of x,y,z coordinates of sites (for which periodic boundary
    conditions may have been applied) that are ordered such that adjacent
    sites in the list should be separated by less than half the simulation
    box lenght and returns their unwrapped (absolute) coords measured position
    relative to the coords of the 1st site in the list
    
    Parameters:
    x,y,z: lists of possibly wrapped coords in x, y and z 
    xboxlo,xboxhi,yboxlo,yboxhi,zboxlo,zboxhi: box boundaries
    
    Returns:
    xu,yu,zu: unwrapped coords in x, y and z
    """
    
    if (len(x) == len(y)) and (len(x) == len(z)):
        nSites = len(x)
    else:
        print("Error: coord arrays in compute_unwrapped_coords unequal in length", file=sys.stderr)
        sys.exit(1)
    
    xbox = xboxhi - xboxlo
    xbox2 = 0.5*xbox
    ybox = yboxhi - yboxlo
    ybox2 = 0.5*ybox
    zbox = zboxhi - zboxlo
    zbox2 = 0.5*zbox
    
    xu = [e for e in x]
    yu = [e for e in y]
    zu = [e for e in z]
    
    xi = xu[0]
    yi = yu[0]
    zi = zu[0]
    
    for j in range(nSites):
        dxji = xu[j] - xi
        dyji = yu[j] - yi
        dzji = zu[j] - zi
        dxji = min_image(dxji,xbox,xbox2)
        dyji = min_image(dyji,ybox,ybox2)
        dzji = min_image(dzji,zbox,zbox2)
        xu[j] = round(xi + dxji,3)
        yu[j] = round(yi + dyji,3)
        zu[j] = round(zi + dzji,3)
        
        xi = xu[j]
        yi = yu[j]
        zi = zu[j]
    # Make sure most sites are inside simulation box:
    x_centre = np.mean(xu)
    y_centre = np.mean(yu)
    z_centre = np.mean(zu)

    if x_centre < xboxlo:
        for i in xu:
            i = i + xbox
    elif x_centre > xboxhi:
        for i in xu:
            i = i - xbox
    if y_centre < yboxlo:
        for i in yu:
            i = i + ybox
    elif y_centre > yboxhi:
        for i in yu:
            i = i - ybox
    if z_centre < zboxlo:
        for i in zu:
            i = i + zbox
    elif z_centre > zboxhi:
        for i in zu:
            i = i - zbox
    return(xu,yu,zu)

def right_handed_coord_system(e_x,e_y,e_z):
    """
    Make a set of 3 vectors assuming to be (x,y,z) arranged right-handed 
    Parameters: 
    e_x, e_y, e_z: column array vectors representing the principal axes of the body
    Returns:
    e_x, e_y, e_z: column array vectors representing the principal axes of the body arranged right-handed
    """
    det_matrix = np.linalg.det(np.concatenate((e_x,e_y,e_z),axis=1))
    if det_matrix < 0:
        e_x = -e_x
    else:
        e_x = e_x
    return(e_x,e_y,e_z)


def global_to_body_transform(R,e_x,e_y,e_z):
    """
    Transformation between global and body frame for a set of 3 vectors 
    that represent principal axes of a CG site. Body frame here is the
    reference CG site
    Parameters:
    R: transformation matrix from global to body frame
    e_x,e_y,e_z: column array vectors representing the principal axes of a CG site
    Returns:
    e_x,e_y,e_z: column array vectors representing the principal axes of a CG site
    in the reference CG site body frame
    """
    e_x = np.dot(R,e_x)
    e_y = np.dot(R,e_y)
    e_z = np.dot(R,e_z)
    return np.concatenate((e_x,e_y,e_z), axis=1)
    
def global_to_body_frame(R,e_x,e_y,e_z):
    """
    Transformation between global and body frame for a set of 3 vectors 
    that represent principal axes of a CG site. Body frame here is the
    reference CG site
    Parameters:
    R: transformation matrix from global to body frame
    e_x,e_y,e_z: column array vectors representing the principal axes of a CG site
    Returns:
    e_x,e_y,e_z: column array vectors representing the principal axes of a CG site
    in the reference CG site body frame
    """
    e_x = np.dot(R,e_x)
    e_y = np.dot(R,e_y)
    e_z = np.dot(R,e_z)
    return (e_x,e_y,e_z)

def global_to_body_vector(R,t,r):
    """
    Transformation between global and body frame for a vector
    This is useful to calculate (x,y,z) coordinate of
    a CG site wrt the reference CG site (to then calculate spherical
    coordinate of that CG site wrt the ref CG site)
    Parameters:
    R: transformation matrix from global to body frame
    t: translation column vector from global to body frame
    r: column array vector representing the coordinate of a CG site in the lab frame
    Returns:
    r: new column array vector representing the coordinate of a CG site
    in the reference body frame
    """
    r = np.dot(R,r+t)
    return r

def pbc_image(xboxlo,xboxhi,yboxlo,yboxhi,zboxlo,zboxhi,site_pos,cutoff):
    """
    Output periodic images. Maximum 3/6 images outputted depending on 
    the site position. This is made to cover non-bonded interactions
    that are caused by pbc  images of the molecules in the simulation box.
    Parameters: xboxlo,xboxhi,yboxlo,yboxhi,zboxlo,zboxhi are box boundaries,
                type int
                site_pos is site position [x,y,z] type "list" 
    Returns: list of positions of images
    """
    #cutoff = 15 # have to add this parameters in later
    xToLo = abs(site_pos[0] - xboxlo)
    xToHi = abs(site_pos[0] - xboxhi)
    yToLo = abs(site_pos[1] - yboxlo)
    yToHi = abs(site_pos[1] - yboxhi)
    zToLo = abs(site_pos[2] - zboxlo)
    zToHi = abs(site_pos[2] - zboxhi)
    site_pos_box = [xToLo,xToHi,yToLo,yToHi,zToLo,zToHi]
    image = []
    num_image = sum(i < cutoff for i in site_pos_box)
    if num_image == 3:
        if xToLo < cutoff:
            if yToLo < cutoff:
                if zToLo < cutoff:
                    image.append([site_pos[0]+xboxhi-xboxlo,site_pos[1],site_pos[2]])
                    image.append([site_pos[0],site_pos[1]+yboxhi-yboxlo,site_pos[2]])
                    image.append([site_pos[0],site_pos[1],site_pos[2]+zboxhi-zboxlo])
                elif zToHi < cutoff:
                    image.append([site_pos[0]+xboxhi-xboxlo,site_pos[1],site_pos[2]])
                    image.append([site_pos[0],site_pos[1]+yboxhi-yboxlo,site_pos[2]])
                    image.append([site_pos[0],site_pos[1],site_pos[2]-(zboxhi-zboxlo)])
            elif yToHi < cutoff:
                if zToLo < cutoff:
                    image.append([site_pos[0]+xboxhi-xboxlo,site_pos[1],site_pos[2]])
                    image.append([site_pos[0],site_pos[1]-(yboxhi-yboxlo),site_pos[2]])
                    image.append([site_pos[0],site_pos[1],site_pos[2]+zboxhi-zboxlo])
                elif zToHi < cutoff:
                    image.append([site_pos[0]+xboxhi-xboxlo,site_pos[1],site_pos[2]])
                    image.append([site_pos[0],site_pos[1]-(yboxhi-yboxlo),site_pos[2]])
                    image.append([site_pos[0],site_pos[1],site_pos[2]-(zboxhi-zboxlo)])
        elif xToHi < cutoff:
            if yToLo < cutoff:
                if zToLo < cutoff:
                    image.append([site_pos[0]-(xboxhi-xboxlo),site_pos[1],site_pos[2]])
                    image.append([site_pos[0],site_pos[1]+yboxhi-yboxlo,site_pos[2]])
                    image.append([site_pos[0],site_pos[1],site_pos[2]+zboxhi-zboxlo])
                elif zToHi < cutoff:
                    image.append([site_pos[0]-(xboxhi-xboxlo),site_pos[1],site_pos[2]])
                    image.append([site_pos[0],site_pos[1]+yboxhi-yboxlo,site_pos[2]])
                    image.append([site_pos[0],site_pos[1],site_pos[2]-(zboxhi-zboxlo)])
            elif yToHi < cutoff:
                if zToLo < cutoff:
                    image.append([site_pos[0]-(xboxhi-xboxlo),site_pos[1],site_pos[2]])
                    image.append([site_pos[0],site_pos[1]-(yboxhi-yboxlo),site_pos[2]])
                    image.append([site_pos[0],site_pos[1],site_pos[2]+zboxhi-zboxlo])
                elif zToHi < cutoff:
                    image.append([site_pos[0]-(xboxhi-xboxlo),site_pos[1],site_pos[2]])
                    image.append([site_pos[0],site_pos[1]-(yboxhi-yboxlo),site_pos[2]])
                    image.append([site_pos[0],site_pos[1],site_pos[2]-(zboxhi-zboxlo)])
    elif num_image == 2:
        if xToLo >= cutoff and xToHi >= cutoff:
            if yToLo < cutoff:
                if zToLo < cutoff:
                    image.append([site_pos[0],site_pos[1]+yboxhi-yboxlo,site_pos[2]])
                    image.append([site_pos[0],site_pos[1],site_pos[2]+zboxhi-zboxlo])
                elif zToHi < cutoff:
                    image.append([site_pos[0],site_pos[1]+yboxhi-yboxlo,site_pos[2]])
                    image.append([site_pos[0],site_pos[1],site_pos[2]-(zboxhi-zboxlo)])
            elif yToHi < cutoff:
                if zToLo < cutoff:
                    image.append([site_pos[0],site_pos[1]-(yboxhi-yboxlo),site_pos[2]])
                    image.append([site_pos[0],site_pos[1],site_pos[2]+zboxhi-zboxlo])
                elif zToHi < cutoff:
                    image.append([site_pos[0],site_pos[1]-(yboxhi-yboxlo),site_pos[2]])
                    image.append([site_pos[0],site_pos[1],site_pos[2]-(zboxhi-zboxlo)])
        elif yToLo >= cutoff and yToHi >= cutoff:
            if xToLo < cutoff:
                if zToLo < cutoff:
                    image.append([site_pos[0]+xboxhi-xboxlo,site_pos[1],site_pos[2]])
                    image.append([site_pos[0],site_pos[1],site_pos[2]+zboxhi-zboxlo])
                elif zToHi < cutoff:
                    image.append([site_pos[0]+xboxhi-xboxlo,site_pos[1],site_pos[2]])
                    image.append([site_pos[0],site_pos[1],site_pos[2]-(zboxhi-zboxlo)])
            elif xToHi < cutoff:
                if zToLo < cutoff:
                    image.append([site_pos[0]-(xboxhi-xboxlo),site_pos[1],site_pos[2]])
                    image.append([site_pos[0],site_pos[1],site_pos[2]+zboxhi-zboxlo])
                elif zToHi < cutoff:
                    image.append([site_pos[0]-(xboxhi-xboxlo),site_pos[1],site_pos[2]])
                    image.append([site_pos[0],site_pos[1],site_pos[2]-(zboxhi-zboxlo)])
        elif zToLo >= cutoff and zToHi >= cutoff:
            if yToLo < cutoff:
                if xToLo < cutoff:
                    image.append([site_pos[0]+xboxhi-xboxlo,site_pos[1],site_pos[2]])
                    image.append([site_pos[0],site_pos[1]+yboxhi-yboxlo,site_pos[2]])
                elif xToHi < cutoff:
                    image.append([site_pos[0]-(xboxhi-xboxlo),site_pos[1],site_pos[2]])
                    image.append([site_pos[0],site_pos[1]+yboxhi-yboxlo,site_pos[2]])
            elif yToHi < cutoff:
                if xToLo < cutoff:
                    image.append([site_pos[0]+xboxhi-xboxlo,site_pos[1],site_pos[2]])
                    image.append([site_pos[0],site_pos[1]-(yboxhi-yboxlo),site_pos[2]])
                elif xToHi < cutoff:
                    image.append([site_pos[0]-(xboxhi-xboxlo),site_pos[1],site_pos[2]])
                    image.append([site_pos[0],site_pos[1]-(yboxhi-yboxlo),site_pos[2]])
    elif num_image == 1:
        if xToLo < cutoff:
            image.append([site_pos[0]+xboxhi-xboxlo,site_pos[1],site_pos[2]])
        elif xToHi < cutoff:
            image.append([site_pos[0]-(xboxhi-xboxlo),site_pos[1],site_pos[2]])
        elif yToLo < cutoff:
            image.append([site_pos[0],site_pos[1]+yboxhi-yboxlo,site_pos[2]])
        elif yToHi < cutoff:
            image.append([site_pos[0],site_pos[1]-(yboxhi-yboxlo),site_pos[2]])
        elif zToLo < cutoff:
            image.append([site_pos[0],site_pos[1],site_pos[2]+zboxhi-zboxlo])
        elif zToHi < cutoff:
            image.append([site_pos[0],site_pos[1],site_pos[2]-(zboxhi-zboxlo)])
    elif num_image == 0:
        image = [0]
    else:
        print("ERROR: Minimum image condition is not apply. \
              More than 3 pbc images are found")
        sys.exit(1)
    return image
