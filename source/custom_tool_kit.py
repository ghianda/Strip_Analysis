import numpy as np
import math


class Bcolors:
    VERB = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def pad_dimension(matrix, shape):
    ''' check if matrix have dimension like 'shape'.
    If not, pad with zero for every axis.'''
    if matrix is not None and shape is not None:

        # check on 0th axis
        if matrix.shape[0] < shape[0]:
            zero_pad = np.zeros((int(shape[0] - matrix.shape[0]), matrix.shape[1], matrix.shape[2]))
            matrix = np.concatenate((matrix, zero_pad), 0)

        # check on 1th axis
        if matrix.shape[1] < shape[1]:
            zero_pad = np.zeros((matrix.shape[0], int(shape[1] - matrix.shape[1]), matrix.shape[2]))
            matrix = np.concatenate((matrix, zero_pad), 1)

        # check on 2th axis
        if matrix.shape[2] < shape[2]:
            zero_pad = np.zeros((matrix.shape[0], matrix.shape[1], int(shape[2] - matrix.shape[2])))
            matrix = np.concatenate((matrix, zero_pad), 2)
        
        return matrix
        
    else:
        raise ValueError('Block or shape is None')


def nextpow2(n):
    return int(np.power(2, np.ceil(np.log2(n))))


def magnitude(x):
    if x is None:
        return magnitude(abs(x))
    elif x == 0:
        return -1
    else:
        return int(math.floor(math.log10(x)))


def normalize_m0_std(mat):
    std = np.std(mat)
    if std != 0:
        # return mat
        return ((mat - np.mean(mat)) / std).astype(np.float32)
    else:
        return (mat - np.mean(mat)).astype(np.float32)


# appendo le coordinate sferiche a quelle cartesiane
def compile_spherical_coord(coord, center, intensity=None):
    # coord : coordinate cartesiane del punto massimo dello spettro, con sistema di riferimento il cubetto stesso
    # center: coordinate del centro del cubetto, con sistema di riferimento il cubetto stesso
    # intensity: valore normalizzato del picco trovato ( misura dell'informazione in frequenza usata per la sogliatura)
    
    # find relative coordinates (center is the (0,0,0) in the relative system) 
    relative = (
        coord[0] - center[0],
        coord[1] - center[1],
        coord[2] - center[2]
    )

    # Spherical coordinates (r, θ, φ) as commonly used in physics (ISO convention): 
    # - radial distance r (rho)
    # - polar angle θ (theta)
    # - and azimuthal angle φ (phi). 

    # z-axis is the optical axis
    
    # phi is the angle on the XY plane. phi = 0 if parallel to y axis (up/down).
    # phi is include in [0, 180] degree if x >= 0 and in (-0, -180) if x<0          
    #    φ (PHI)
    #                    
    #                  
    #            -180|+180			xy plane
    #         IV     |		I
    #                |
    #      -90       |        90
    #      ----------0----------    x>0
    #                |
    #         III    |		II
    #                |
    #                |0
    #				y>0

    # 	θ (THETA) is the angle down by the z axis. theta = 0 if parallel to z axis, and theta = 90 if parallel to xy plane. 
    # theta is include in [0, 90] degree if z >= 0 and in (90, 180] if z<0    
    #          
    #    THETA
    #                     
    #                  
    #                |+180		(xz or yz view)
    #          IV    |		I
    #                |
    #      +90       |       +90
    #      ----------0----------    xy plane
    #                |
    #          III   |		II
    #                |
    #                |0
    # 				z>0

    # find spherical coordinates (OLD)
    spherical = np.zeros(len(coord))
    xy = relative[0]**2 + relative[1]**2
    spherical[0] = np.sqrt(xy + relative[2]**2)  # radius
    spherical[1] = (180 / np.pi) * np.arctan2(np.sqrt(xy), relative[2])  # for elevation angle defined from Z-axis down ()
    #spherical[1] = (180 / np.pi) * np.arctan2(xyz[2], np.sqrt(xy))  # for elevation angle defined from XY-plane up
    spherical[2] = (180 / np.pi) * np.arctan2(relative[1], relative[0]) # phi
    
    # NEW
    # spherical = np.zeros(len(coord))
    # xy = relative[0]**2 + relative[1]**2
    # xr_abs = np.abs(relative[0])
    # yr_abs = np.abs(relative[1])
    
    # spherical[0] = np.sqrt(xy + relative[2]**2)  # radius (rho)
    # spherical[1] = (180 / np.pi) * np.arctan2(np.sqrt(xy), relative[2])  # theta - for elevation angle defined from Z-axis down ()
    # #spherical[1] = (180 / np.pi) * np.arctan2(xyz[2], np.sqrt(xy))  # theta - for elevation angle defined from XY-plane up
    # spherical[2] = (180 / np.pi) * ((np.pi / 2) - np.arctan2(yr_abs, xr_abs)) # phi
    

    complete_coord = {
        'cartesian' : tuple(coord),      # x, y, z
        'relative to center'  : tuple(relative),   # xr, yr, zr
        'spherical' : {
            'val' : tuple(spherical),   # rho, phi, theta
            'legend' : ('rho', 'theta', 'phi')},
        'intensity' : intensity
    }
    return complete_coord


# calcola solo le cartesiane
def spherical_coord(coord, center):
    # coord : coordinate cartesiane del punto massimo dello spettro, con sistema di riferimento il cubetto stesso
    # center: coordinate del centro del cubetto, con sistema di riferimento il cubetto stesso
    
    # find relative coordinates (center is the (0,0,0) in the relative system) 
    relative = (
        coord[0] - center[0],
        coord[1] - center[1],
        coord[2] - center[2]
    )

    # Spherical coordinates (r, φ, θ) as commonly used in physics (ISO convention): 
    # - radial distance r (rho)
    # - polar angle θ (theta)
    # - and azimuthal angle φ (phi). 

    # z-axis is the optical axis
    
    # phi is the angle on the XY plane. phi = 0 if parallel to y axis (up/down).
    # phi is include in [0, 180] degree if x >= 0 and in (-0, -180) if x<0          
    #    φ (PHI)
    #                    
    #                  
    #            -180|+180			xy plane
    #         IV     |		I
    #                |
    #      -90       |        90
    #      ----------0----------    x>0
    #                |
    #         III    |		II
    #                |
    #                |0
    #				y>0

    # 	θ (THETA) is the angle down by the z axis. theta = 0 if parallel to z axis, and theta = 90 if parallel to xy plane. 
    # theta is include in [0, 90] degree if z >= 0 and in (90, 180] if z<0    
    #          
    #    THETA
    #                     
    #                  
    #                |+180		(xz or yz view)
    #          IV    |		I
    #                |
    #      +90       |       +90
    #      ----------0----------    xy plane
    #                |
    #          III   |		II
    #                |
    #                |0
    # 				z>0
    
    # find spherical coordinates
    spherical = np.zeros(len(coord))
    xy = relative[0]**2 + relative[1]**2
    
    radius = np.sqrt(xy + relative[2]**2) 
    theta = (180 / np.pi) * np.arctan2(np.sqrt(xy), relative[2])  # theta - for elevation angle defined from Z-axis down to xy plane
    phi = (180 / np.pi) * np.arctan2(relative[1], relative[0]) # phi

    return (radius, phi, theta)


def search_value_in_txt(filepath, strings_to_search):
    # strings_to_search is a string or a list of string
    if type(strings_to_search) is not list:
        strings_to_search = [strings_to_search]
        
    # read all words in filepath
    words = all_words_in_txt(filepath)
    
    # search strings
    values = [words[words.index(s) + 2] for s in strings_to_search if s in words]
    
    return values


def write_on_txt(strings, txt_path, _print=False, mode='a'):
    # write the lines in 'strings' list into .txt file addressed by txt_path
    # if _print is True, the lines is printed
    #
    with open(txt_path, mode=mode) as txt:
        for s in strings:
            txt.write(s + '\n')
            if _print:
                print(s)


def all_words_in_txt(filepath):
    words = list()
    with open(filepath, 'r') as f:
        data = f.readlines()
        for line in data:
            for word in line.split():
                words.append(word)
    return words


def create_slice_coordinate(start_coord, shape_of_subblock):
    # create slice coordinates for take a submatrix with shape = shape_of_subblock
    # that start at start_coord
    selected_slice_coord = []
    for (start, s) in zip(start_coord, shape_of_subblock):
        selected_slice_coord.append(slice(start, start + s, 1))  # (start, end, step=1)
    return selected_slice_coord


def create_coord_by_iter(r, c, z, shape_P, _z_forced=False):
    # create init coordinate for parallelepiped

    row = r * shape_P[0]
    col = c * shape_P[1]

    if _z_forced:
        zeta = z
    else:
        zeta = z * shape_P[2]

    return (row, col, zeta)


def seconds_to_hour_min_sec(sec):
    # convert seconds to (hour, minutes, seconds)

    if sec < 60:
        return int(0), int(0), int(sec)
    elif sec < 3600:
        return int(0), int(sec // 60), int(sec % 60)
    else:
        h = int(sec // 3600)
        m = int(sec // 60) - (h * 60)
        return h, m, int(sec % 60)


def print_time_execution(start, end, proc_name='Process'):
    h, m, s = seconds_to_hour_min_sec(end - start)
    print(proc_name,'Executed in: {}h:{}m:{}s'.format(h, m, s))
    return None





