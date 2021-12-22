# script che estrare la distribuzione dell'Allineamento (inteso come all. all'asse Y) (comp. y di ogni vettore -> media)
# da una matrice R già esistente.

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

# R, matrix_of_align_perc, shape_G, isolated_value = estimate_local_alignment(R, parameters)
def estimate_local_alignment(R, parameters, grane=None, neighbours_lim=None):
    '''
        :param R: Result matrix 'R'
        :param parameters: dictionary of parameters read from parameters.txt
        :return: Result matrix 'R' (matrix of orientations)
        :return shape_G (dimension of grane of local alignment analysis
        :return isolated value (value to assign at isolated points'''

    """ Calculate and save inside a numpy 3d matrix the 'local_alignment' - a float value between [0, 100] if valid, or -1 if not valid (read above):
    0   : min alignment (all local vectors are orthogonal to  Y axis)
    100 : max alignment (all local vectors are aligned with Y axis)
    -1  : too many neighbours is not valid (freq_info == False) -> block is isolated (not valid)

    local_alignement := average of Y-components (first axis) of the vectors

    SubBlock (Grane of analysis) dimension for alignment estimation (if not passed) have shape = shape_G = (grane_size_xy, grane_size_xy, grane_size_z)
    with grane_size_xy and grane_size_z readed from parameters.txt file.

    Condition:
    1) if grane_size_xy or grane_size_z < 2, function use 2.
    2) if inside a SubBlock there is less than valid peak than 'lim_on_local_dispersion_eval' parameters value, 
       local_alignment is setted to -1 (isolated block). 
       For visualization, these blocks are set with a predefined value (for example, 0)."""

    res_xy = parameters['res_xy']
    res_z = parameters['res_z']
    num_of_slices_P = parameters['num_of_slices_P']
    resolution_factor = res_z / res_xy
    block_side = int(num_of_slices_P * resolution_factor)

    # Pixel size in um^-1
    pixel_size_F_xy = 1 / (block_side * res_xy)
    pixel_size_F_z = 1 / (num_of_slices_P * res_z)

    # read isolated value from parameters
    isolated_value = parameters['isolated']

    if grane is None:
        # extract analysis subblock dimension from parameters
        grane_size_z = parameters['local_disarray_z_side']
        grane_size_xy = parameters['local_disarray_xy_side']
    else:
        grane_size_xy = grane_size_z = grane

    # shape of grane of analysis
    shape_G = (int(grane_size_xy), int(grane_size_xy), int(grane_size_z))

    # define limimt of number of vectors in the disarray macrovoxel
    if parameters['neighbours_lim'] > 0:
        neighbours_lim = parameters['neighbours_lim']  # manually inserted from parameters.txt
    else:
        neighbours_lim = np.int(np.prod(shape_G) / 2)  # auto: 50% of vectors

    # iteration long each axis (ceil -> upper integer)
    iterations = tuple(np.ceil(np.array(R.shape) / np.array(shape_G)).astype(np.uint32))
    # print('iterations:', iterations)

    # define global matrix that contains each local disarray
    matrix_of_align_perc = np.zeros(iterations).astype(np.float32)

    # counter
    _i = 0

    for z in range(iterations[2]):
        for r in range(iterations[0]):
            for c in range(iterations[1]):

                # grane extraction from R
                start_coord = create_coord_by_iter(r, c, z, shape_G)
                slice_coord = create_slice_coordinate(start_coord, shape_G)
                macrovoxel = R[slice_coord]

                # select only blocks with valid frequency information
                f_map = macrovoxel['freq_info']
                grane_f = macrovoxel[f_map].ravel()

                # check if grane_f have at least neighbours_lim elements (default: 3)
                if grane_f.shape[0] >= neighbours_lim:

                    # vector components (as a N x 3 matrix) : N = grane_f.shape[0] = number of valid blocks
                    coord = grane_f['quiver_comp'][:, 0, :]

                    # resolution: from pixel to um-1
                    coord_um = coord * np.array([pixel_size_F_xy, pixel_size_F_xy, pixel_size_F_z])

                    # normalize vectors (every row is a 3D vector):
                    coord_um_norm = coord_um / np.linalg.norm(coord_um, axis=1).reshape(coord_um.shape[0], 1)

                    # mean of the Y-component of each vectors (between [0, 1])
                    # NB: absolute values because I assume all the vectors in the Y-axis positive direction
                    y_mean = np.mean(np.abs(coord_um_norm[:, 0]))  # [all_vectors, first axis]

                    # define local_alignment degree
                    local_alignment_perc = 100 * y_mean

                    # save it in the matrix of local alignment (perc)
                    matrix_of_align_perc[r, c, z] = local_alignment_perc

                else:
                    # there are too few vectors in this grane
                    matrix_of_align_perc[r, c, z] = -1

    return R, matrix_of_align_perc, shape_G, isolated_value


def main(parser):

    # read args from console
    args = parser.parse_args()
    R_filepath = args.source_R
    parameters_filename = args.source_P

    # extract paths and filenames
    base_path           = os.path.dirname(R_filepath)
    Parameters_filepath = os.path.join(base_path, parameters_filename)
    R_filename          = os.path.basename(R_filepath)
    R_prefix            = R_filename.split('.')[0]

    # reads parameters
    parameters = extract_parameters(Parameters_filepath)

    # load R numpy file
    R = np.load(R_filepath)
    print('R loaded with shape (r, c, z): ', R.shape)
    print('\nR_dtype: \n', R.dtype)

    # Estimate local Alignment (matrix_of_alignment) and save it in a numpy file
    R, matrix_of_align_perc, shape_G, isolated_value = estimate_local_alignment(R, parameters)

    print('Local Alignment estimated inside result Matrix')
    print(' - grane (r, c, z) used: ({}, {}, {})'.format(shape_G[0], shape_G[1], shape_G[2]))
    print(' - isolated values set at {}'.format(-1))

    # save disarray matrix in a numpy file
    mtrx_fname = 'Alignment_matrix_g{}x{}x{}_of_{}.npy'.format(shape_G[0], shape_G[1], shape_G[2], R_prefix)
    np.save(file=os.path.join(base_path, mtrx_fname), arr=matrix_of_align_perc)
    print('Alignment matrix saved as ', mtrx_fname)
    print('in: \n', base_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimation of Local Alignment from Orientation vectors (R matrix)')
    parser.add_argument('-r', '--source-R', help='Filepath of the Orientation matrix to load', required=True)
    parser.add_argument('-p', '--source-P', help='Filename of Parameters .txt file to load',
                        default='parameters.txt', required=False)
    main(parser)
