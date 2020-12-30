# script che elabora il disarray (vettore locale medio -> modulo -> (1 - modulo)
# da una matrice R già esistente.
# NOTA BENE - è fatto per le matrici R delle strip - allora il campo dentro R si chiam local_disordersi chiamano DISORDER

import numpy as np
import argparse
import os


def create_coord_by_iter(r, c, z, shape_P, _z_forced=False):
    # create init coordinate for parallelepiped

    row = r * shape_P[0]
    col = c * shape_P[1]

    if _z_forced:
        zeta = z
    else:
        zeta = z * shape_P[2]

    return (row, col, zeta)


def create_slice_coordinate(start_coord, shape_of_subblock):
    # create slice coordinates for take a submatrix with shape = shape_of_subblock
    # that start at start_coord
    selected_slice_coord = []
    for (start, s) in zip(start_coord, shape_of_subblock):
        selected_slice_coord.append(slice(start, start + s, 1))  # (start, end, step=1)
    return selected_slice_coord


def extract_parameters(filename):
    ''' read parameters values in filename.txt
    and save it in a dictionary'''

    param_names = ['num_of_slices_P',
                   'sarc_length',
                   'res_xy', 'res_z',
                   'threshold_on_cell_ratio',
                   'threshold_on_peak_ratio',
                   'threshold_on_hyperbole',
                   'psd0_hyperbole_psd_ratio',
                   'k_hyperbole_psd_ratio',
                   'y0_hyperbole_psd_ratio',
                   'sigma',
                   'local_disarray_xy_side',
                   'local_disarray_z_side',
                   'neighbours_lim',
                   'isolated']

    # read values in txt
    param_values = search_value_in_txt(filename, param_names)

    print('\n ***  Parameters : \n')
    # create dictionary of parameters
    parameters = {}
    for i, p_name in enumerate(param_names):
        parameters[p_name] = float(param_values[i])
        print(' - {} : {}'.format(p_name, param_values[i]))
    print('\n \n')
    return parameters


def all_words_in_txt(filepath):
    words = list()
    with open(filepath, 'r') as f:
        data = f.readlines()
        for line in data:
            for word in line.split():
                words.append(word)
    return words


def search_value_in_txt(filepath, strings_to_search):
    # strings_to_search is a string or a list of string
    if type(strings_to_search) is not list:
        strings_to_search = [strings_to_search]

    # read all words in filepath
    words = all_words_in_txt(filepath)

    # search strings
    values = [words[words.index(s) + 2] for s in strings_to_search if s in words]

    return values


def estimate_local_disarry(R, parameters):
    '''
        :param R: Result matrix 'R'
        :param parameters: dictionary of parameters read from parameters.txt
        :return: Result matrix 'R' with 'local_disorder' value saved inside every valid cell (with freq_info == True
        :return shape_G (dimension of grane os local disorder analysis
        :return isolated value (value to assign at isolated points'''

    """ Calculate and save inside 'local_disorder' a float value between [0, 1] if valid, or -1 if not valid (read above):
    0 : max order (all neighbour has same direction)
    1 : max disorder (neighbour with orthogonal direction)
    -1: too many neighbours is not valid (freq_info == False) -> block is isolated (not valid)

    local_disorder := module of std. deviation (3, 1) array (for every dimension (r, c, z),
    the std. dev. of peak components of neighbour)

    SubBlock (Grane of analysis) dimension for std. dev. estimation have shape = shape_G = (grane_size_xy, grane_size_xy, grane_size_z)
    with grane_size_xy and grane_size_z readed from parameters.txt file.

    Condition:
    1) if grane_size_xy or grane_size_z < 2, function use 2.
    2) if inside a SubBlock there is less than valid peak than 'lim_on_local_dispersion_eval' parameters value, local_disorder is setted to -1 (isolated block)
       for visualization, these blocks are setted with maximum local_disorder value."""

    res_xy = parameters['res_xy']
    res_z = parameters['res_z']
    num_of_slices_P = parameters['num_of_slices_P']
    resolution_factor = res_z / res_xy
    block_side = int(num_of_slices_P * resolution_factor)

    #       Pixel size in um^-1
    pixel_size_F_xy = 1 / (block_side * res_xy)
    pixel_size_F_z = 1 / (num_of_slices_P * res_z)

    neighbours_lim = parameters['neighbours_lim'] if parameters['neighbours_lim'] > 3 else 3
    print('neighbours_lim', neighbours_lim)

    # extract analysis subblock dimension from parameters
    grane_size_z = parameters['local_disarray_z_side'] if parameters['local_disarray_z_side'] > 2 else 2
    grane_size_xy = parameters['local_disarray_xy_side']

    # check if value is valid
    if grane_size_xy == 0:
        grane_size_xy = grane_size_z * resolution_factor
    elif grane_size_xy < 2:
        grane_size_xy = 2

    # print('grane_size_z', grane_size_z)
    # print('grane_size_z', grane_size_xy)

    # shape of grane of analysis
    shape_G = (int(grane_size_xy), int(grane_size_xy), int(grane_size_z))
    # print('shape_G', shape_G)

    # iteration long each axis (ceil -> upper integer)
    iterations = tuple(np.ceil(np.array(R.shape) / np.array(shape_G)).astype(np.uint32))
    print('iterations', iterations)

    # define global matrix that contains each local disarray
    matrix_of_disarray_perc = np.zeros(iterations).astype(np.float32)

    # counter
    _i = 0

    for z in range(iterations[2]):
        for r in range(iterations[0]):
            for c in range(iterations[1]):
                print('.', end='')
                # _i += 1

                # grane extraction from R
                start_coord = create_coord_by_iter(r, c, z, shape_G)
                slice_coord = create_slice_coordinate(start_coord, shape_G)
                grane = R[slice_coord]

                # select only blocks with valid frequency information
                f_map = grane['freq_info']
                grane_f = grane[f_map]

                # check if grane_f have at least neighbours_lim elements (default: 3)
                if grane_f.shape[0] >= neighbours_lim:

                    # vector components (as a N x 3 matrix) : N = grane_f.shape[0] = number of valid blocks
                    coord = grane_f['quiver_comp'][:, 0, :]

                    # resolution: from pixel to um-1
                    coord_um = coord * np.array([pixel_size_F_xy, pixel_size_F_xy, pixel_size_F_z])

                    # normalize vectors (every row is a 3D vector):
                    coord_um_norm = coord_um / np.linalg.norm(coord_um, axis=1).reshape(coord_um.shape[0], 1)

                    # take a random versor (for example, the first)
                    v1 = coord_um_norm[0, :]

                    # move all the vectors in the same direction of v1
                    # (checking the positive or negative result of dot product
                    # between the v1 and the others)
                    for i in range(coord_um_norm.shape[0]):
                        scalar = np.dot(v1, coord_um_norm[i])
                        if scalar < 0:
                            # change the direction of i-th versor
                            coord_um_norm[i] = coord_um_norm[i] * -1

                    # local alignment degree: module of the average vector
                    alignment = np.linalg.norm(np.average(coord_um_norm, axis=0))

                    # define local_disarray degree
                    local_disarray_perc = 100 * (1 - alignment)

                    # save it in each block of this portion (grane) for future statistics and plot
                    R[slice_coord]['local_disorder'] = local_disarray_perc

                    # and save it in the matrix of local_disarray
                    matrix_of_disarray_perc[r, c, z] = local_disarray_perc

                else:
                    # there are tto few vectors in this grane
                    R[slice_coord]['local_disorder'] = -1.  # assumption that isolated quiver have no disarray
                    matrix_of_disarray_perc[r, c, z] = -1


    # read isolated value from parameters and normalize values inside R between 0 and 1
    isolated_value = parameters['isolated']
    # R = normalize_local_disorder(R, max_dev, min_dev, isolated_value)
    return R, matrix_of_disarray_perc, shape_G, isolated_value

def main(parser):

    # read args from console
    args = parser.parse_args()
    R_filepath = args.source_R[0]
    parameters_filename = args.source_P[0]

    # extract paths and filenames
    base_path = os.path.dirname(R_filepath)
    Parameters_filepath = os.path.join(base_path, parameters_filename)
    R_filename = os.path.basename(R_filepath)
    R_prefix = R_filename.split('.')[0]

    # reads parameters
    parameters = extract_parameters(Parameters_filepath)

    # load R numpy file
    R = np.load(R_filepath)
    print('R loaded with shape (r, c, z): ', R.shape)
    print('\nR_dtype: \n', R.dtype)

    # Estimate local Disarray (matrix_of_disarray) and save it in a numpy file and inside R
    R, matrix_of_disarray_perc, shape_G, isolated_value = estimate_local_disarry(R, parameters)

    print('Local Disarray estimated inside result Matrix')
    print(' - grane (r, c, z) used: ({}, {}, {})'.format(shape_G[0], shape_G[1], shape_G[2]))
    print(' - isolated values set at {}'.format(isolated_value))

    # save disarray matrix in a numpy file
    disarray_fname = 'Disarray_matrix_g{}x{}x{}_of_{}.npy'.format(shape_G[0], shape_G[1], shape_G[2], R_prefix)
    np.save(file=os.path.join(base_path, disarray_fname), arr=matrix_of_disarray_perc)
    print('Disarray matrix saved as ', disarray_fname)
    print('in: \n', base_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimation of Local Disarray from Orientation vectors')
    parser.add_argument('-r', '--source-R', nargs='+', help='Filepath of the Orientation matrix to load', required=False)
    parser.add_argument('-p', '--source-P', nargs='+', help='Filename of Parameters .txt file to load', required=False)
    main(parser)
