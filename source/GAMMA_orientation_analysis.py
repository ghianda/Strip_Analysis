# general
import argparse
import numpy as np
import os
import time

# specific
from scipy.ndimage.filters import gaussian_filter

# custom codes
from make_data import load_stack_into_numpy_ndarray, manage_path_argument
from custom_image_tool import normalize
from custom_freq_analysis import create_3D_filter, fft_3d_to_cube, find_centroid_of_peak, filtering_3D, find_peak_in_psd
from custom_tool_kit import pad_dimension, create_slice_coordinate, search_value_in_txt, spherical_coord, seconds_to_min_sec

# --------------------------------------- end import --------------------------------------------


def main(parser):
    ''' Docstring (TODO) parameters '''

    # ====== 0 ====== Initial operations
    # TODO estrare dai parametri da terminale la bool 'verbose'
    verbose = False

    # read args from console
    args = parser.parse_args()
    source_path = manage_path_argument(args.source_folder)


    # source_path = args.source_folder

    # if type(source_path) is list:
    #     if len(source_path) > 1:
    #         given_path = ' '.join(source_path)
    #         source_path = [given_path]  # prepare source_path for 'make_dataset' function (it takes a list in input)
    #     else:
    #         given_path = source_path[0]
    # else:
    #     given_path = source_path

    # # remove last backlslash
    # if given_path.endswith('/'):
    #     given_path = given_path[0:-1]

    # take base path and stack name
    base_path = os.path.dirname(os.path.dirname(source_path))
    stack_name = os.path.basename(source_path)

    # Def .txt filepath
    txt_parameters_path = os.path.join(base_path, 'parameters.txt')
    txt_results_path = os.path.join(base_path, 'GAMMA_orientation_results.txt')


    Results_filename = 'orientation_Results.npy'
    Results_filepath = os.path.join(base_path, Results_filename)

    # print to video and write in results.txt init message
    init_message = [
        ' *****************   GAMMA - Orientation analysis of 3D stack   *****************\n'
        ' Source from path : {}'.format(source_path),
        ' Base path : {}'.format(base_path),
        ' Stack : {}'.format(stack_name),
    ]

    with open(txt_results_path, 'w') as f:
        for line in init_message:
            print(line)
            f.write(line + '\n')

    # reads parameters
    parameters = extract_parameters(txt_parameters_path)

    # analysis block dimension in z-axis
    num_of_slices_P = parameters['num_of_slices_P']

    # Parameters of Acquisition System:
    res_z = parameters['res_z']
    res_xy = parameters['res_xy']
    resolution_factor = res_z / res_xy
    block_side = row_P = col_P = int(num_of_slices_P * resolution_factor)
    shape_P = np.array((row_P, col_P, num_of_slices_P)).astype(np.int32)

    """ =====================================================================================
    __________________________  -1-  OPEN STACK _______________________________________________"""

    loading_mess = list()
    loading_mess.append(' ***** Start load the Stack, this may take a few minutes... ')
    
    # extract data
    volume, message = load_stack_into_numpy_ndarray([source_path])
    loading_mess.append(message)
    # source_data = make_dataset(source_path)
    # data_length = len(source_data)

    # volume = create_stack_light(source_data)  # NB - this function delete source_data items and move data inside volume
    # del source_data

    # if data_length == volume.shape[2]:
    #     print('   OK : {} slices loaded'.format(volume.shape[2]))
    # else:
    #     print(' *** WARNING -> len(image_list) != slices in ndarray -> check loading')
    
    loading_mess.append(' - Volume shape : {}'.format(volume.shape))
    with open(txt_results_path, 'a') as f:
        for m in loading_mess:
        	print(m)
        	f.write(m + '\n')

    # calculate dimension
    shape_V = np.array(volume.shape)
    pixel_for_slice = shape_V[0] * shape_V[1]

    """ =====================================================================================
    ________________________  -2-   CYCLE FOR BLOCKS EXTRACTION and ANALYSIS __________________"""
    t_start = time.time()

    # create empty Result matrix
    R, shape_R = create_R(shape_V, shape_P)

    # create 3D filter
    mask = create_3D_filter(block_side, res_xy, parameters['sarc_length'])

    count = 1  # count iteration
    total_iter = np.prod(shape_R)
    print('\n \n ***** Start iteration of analysis, expectd iterations : {} \n'.format(total_iter))

    with open(txt_results_path, 'a') as f:
        for z in range(shape_R[2]):
            for r in range(shape_R[0]):
                for c in range(shape_R[1]):
                    # initialize list of string lines
                    lines = []

                    start_coord = create_coord_by_iter(r, c, z, shape_P)
                    slice_coord = create_slice_coordinate(start_coord, shape_P)
                    if verbose: lines.append('\n \n')
                    lines.append('- iter: {} - init_coord : {} - on total: {}'.format(count, start_coord, total_iter))

                    # save init info in R
                    R[r, c, z]['id_block'] = count
                    R[r, c, z]['init_coord'] = start_coord

                    # extract parallelepiped
                    parall = volume[slice_coord]

                    # check dimension (if iteration is on border of volume, add zero_pad)
                    parall = pad_dimension(parall, shape_P)

                    if np.max(parall) != 0:
                        parall = (normalize(parall)).astype(np.float32)  # fft analysis work with float

                        # analysis of parallelepiped extracted
                        there_is_cell, there_is_freq, results = block_analysis(
                                                                                parall,
                                                                                shape_P,
                                                                                parameters,
                                                                                block_side,
                                                                                mask,
                                                                                verbose,
                                                                                lines)
                        # save info in R[r, c, z]
                        if there_is_cell: R[r, c, z]['cell_info'] = True
                        if there_is_freq: R[r, c, z]['freq_info'] = True

                        # save results in R
                        for key in results.keys(): R[r, c, z][key] = results[key]

                    else:
                        if verbose: lines.append('   block rejected')

                    for l in lines: print(l)
                    count += 1

    # execuiton time
    (h, m, s) = seconds_to_min_sec(time.time() - t_start)
    print('\n Iterations ended successfully \n')

    """ =====================================================================================
        ________________________  -3-   RESULTS ANALYSIS   __________________________________"""

    post_proc_mess = list()

    # count results, rejected and accepted blocks
    block_with_cell = np.count_nonzero(R['cell_info'])
    block_with_peak = np.count_nonzero(R['freq_info'])
    p_rejec_cell = 100 * (1 - block_with_cell / count)
    p_rejec_freq_tot = 100 * (1 - block_with_peak / count)
    p_rejec_freq = 100 * (1 - block_with_peak / block_with_cell)

    post_proc_mess.append('\n ***** End of iterations, time of execution: {0:2d}h {1:2d}m {2:2d}s \n'.format(int(h), int(m), int(s)))
    post_proc_mess.append('\n - Expected iterations : {}'.format(total_iter))
    post_proc_mess.append(' - Total iterations : {}'.format(count - 1))
    post_proc_mess.append('\n - block with cell : {}, rejected from total: {} ({}%)'.format(block_with_cell,
                                                                                                count - block_with_cell,
                                                                                                p_rejec_cell))
    post_proc_mess.append(' - block with freq. info : {}'
                           '\n    rejected from total: {} ({}%)'
                           '\n    rejected from block with cell: {} ({}%)'.format(block_with_peak,
                                                                                  count - block_with_peak,
                                                                                  p_rejec_freq_tot,
                                                                                  block_with_cell - block_with_peak,
                                                                                  p_rejec_freq))
    with open(txt_results_path, 'a') as f:
        for m in post_proc_mess:
            print(m)
            f.write(m + '\n')

    post_proc_mess = list()

    # threshold results on frequency validation parameter and save matrix
    mess = '\n \n *** Analysis of Results : remove block with low frequency affidability \n'
    post_proc_mess.append(mess)
    print(mess)

    # - 1 normalization of psd_ratio values
    R = parameter_normalizer(R, 'psd_ratio')
    mess = '- 1 - Normalization on \'psd_ratio\': complete'
    post_proc_mess.append(mess)
    print(mess)

    # - 2 thresholding on psd_ratio values
    R, before, after = threshold_par(R, parameters, 'psd_ratio')
    mess = '- 2 - First thresholding based on PSD Information: selected {} blocks from {}'.format(after, before)
    post_proc_mess.append(mess)
    print(mess)

    # - 3 outlier remotion based of orientation and psd_ratio values
    R, before, after = remove_outlier(R, parameters, 'psd_ratio')
    mess = '- 3 - Outlier Remotion based on PSD Information: removed {} outlier from {} blocks. True blocks: {}'\
            .format(before - after, before, after)
    post_proc_mess.append(mess)
    print(mess)

    # save Result matrix
    np.save(Results_filepath, R)

    # - 4 Estimate and write local disorder inside Result Matrix
    R, shape_LD, isolated_value = estimate_local_disorder(R, parameters, resolution_factor)
    mess = '- 4 - Local Disorder estimated inside result Matrix, with grane (r, c, z): ({}, {}, {}) ' \
           'and isolated points setted with local_disorder = {}'\
            .format(shape_LD[0], shape_LD[1], shape_LD[2], isolated_value)
    post_proc_mess.append(mess)
    print(mess)

    # se funziona tutto, salvare solo una versione di R
    Results_filename = 'orientation_Results_after_disorder.npy'
    Results_filepath = os.path.join(base_path, Results_filename)
    np.save(Results_filepath, R)

    with open(txt_results_path, 'a') as f:
        for m in post_proc_mess:
            f.write(m + '\n')
    del post_proc_mess

    """ =====================================================================================
            ________________________  -4-   STATISTICS   __________________________________"""

    stat = statistics(R, parameters)
    result_mess = list()
    result_mess.append('\n \n *** Results of statistical analysis on accepted points: \n')
    result_mess.append(' - {0} : {1:.3f} um^(-1)'.format('Mean module', stat['Mean Module']))
    result_mess.append(' - {0} : {1:.3f} um'.format('Mean Period', stat['Mean Period']))
    result_mess.append(' - {0} : {1:.3f} % '.format('Alignment', 100 * stat['Alignment']))
    result_mess.append(' - {0} : {1:.3f} % '.format('XZ Dispersion (area_ratio)', 100 * stat['area_ratio']))
    result_mess.append(' - {0} : {1:.3f} % '.format('XZ Dispersion (sum dev.std)', 100 * stat['sum_std']))
    result_mess.append(' \n \n ***************************** END GAMMA - orientation_analysis.py ********************'
                       '\n \n \n \n ')

    with open(txt_results_path, 'a') as f:
        for l in result_mess:
            print(l)
            f.write(l + '\n')

    # -------------------------------------- end main -----------------------------------------------


def estimate_local_disorder(R, parameters, resolution_factor):
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

    SubBlock (Grane of analysis) dimension for std. dev. estimation have shape = shape_G = (Ng_xy, Ng_xy, Ng_z)
    with Ng_xy and Ng_z readed from parameters.txt file.

    Condition:
    1) if Ng_xy or Ng_z < 2, function use 2.
    2) if inside a SubBlock there is less than valid peak than 'lim_on_local_dispersion_eval' parameters value, local_disorder is setted to -1 (isolated block)
       for visualization, these blocks are setted with maximum local_disorder value."""

    neighbours_lim = parameters['neighbours_lim'] if parameters['neighbours_lim'] > 3 else 3

    # extract analysis subblock dimension from parameters
    Ng_z = parameters['local_disorder_z_side'] if parameters['local_disorder_z_side'] > 2 else 2
    Ng_xy = parameters['local_disorder_xy_side']

    # check if value is valid
    if Ng_xy == 0:
        Ng_xy = Ng_z * resolution_factor
    elif Ng_xy < 2:
        Ng_xy = 2

    # shape of grane of analysis
    shape_G = (int(Ng_xy), int(Ng_xy), int(Ng_z))

    # iteration long each axis
    iterations = tuple(np.ceil(np.array(R.shape) / np.array(shape_G)).astype(np.uint32))

    max_dev = min_dev = 0

    for z in range(iterations[2]):
        for r in range(iterations[0]):
            for c in range(iterations[1]):

                # grane extraction from R
                start_coord = create_coord_by_iter(r, c, z, shape_G)
                slice_coord = create_slice_coordinate(start_coord, shape_G)
                grane = R[slice_coord]

                # takes only block with valid frequency information
                f_map = grane['freq_info']
                grane_f = grane[f_map]

                # check if grane_f have at least neighbours_lim elements (default: 3)
                if grane_f.shape[0] > neighbours_lim:

                    # estraggo le componenti dei vettori picchi
                    coord = np.vstack((grane_f['quiver_comp'][:, 0, 0],
                                       grane_f['quiver_comp'][:, 0, 1],
                                       grane_f['quiver_comp'][:, 0, 2]))

                    # calcolo somma in quadratura delle tre deviazione standard delle tre componenti
                    dev_rcz = np.std(coord, axis=1)
                    dev_tot = np.linalg.norm(dev_rcz)

                    # salvo in R
                    R[slice_coord]['local_disorder'] = dev_tot  # dev_tot

                    # update maximun and minimum value finded
                    max_dev = dev_tot if dev_tot > max_dev else max_dev
                    min_dev = dev_tot if dev_tot < min_dev else min_dev


                else:
                    R[slice_coord]['local_disorder'] = -1.  # assumption that isolated quiver have no caos

    # read isolated value from parameters and normalize values inside R between 0 and 1
    isolated_value = parameters['isolated']
    R = normalize_local_disorder(R, max_dev, min_dev, isolated_value)
    return R, shape_G, isolated_value


def normalize_local_disorder(R, max_dev, min_dev, isolated_value):
    ''' First swap '-1' values with max_dev or min_dev (choice by external parameters),
    then normalize all values between 0 and 1 (0 max order, 1 max disorder)
    :param R : Result matrix
    :param max_dev : max value of local_disorder attribute in R
    :param min_dev : min value of local_disorder attribute in R
    :param isolated_value : 1 or 0 (max disorder or max order), value to assign at isolated blocks'''

    shape_R = R.shape
    diff = max_dev - min_dev

    for z in range(shape_R[2]):
        for r in range(shape_R[0]):
            for c in range(shape_R[1]):
                if R[r, c, z]['local_disorder'] == -1:
                    R[r, c, z]['local_disorder'] = isolated_value
                else:
                    R[r, c, z]['local_disorder'] = (R[r, c, z]['local_disorder'] - min_dev) / diff
    return R


def statistics(R, parameters):
    ''' From parameters read and calculate parameters of Acquisition System
    ( like pixel_sizes, dimensions)
    From R take coordinates of Peaks from blocks that have frequency validation

    Change coordinates of peaks (relative to center of block) and mMove peaks in Y > 0 subspace
    (because to find peak with y>0 or peak with y<0 in spectrum is random : there are symmetrical

    Estimate
    - mean module of vectors -> Frequency of sarcomeres pattern
    - mean Y components normalized between 0 and 1 ->  alignement
    - std. deviation of X and Z components  ->  angular dispersion (disorder) on XZ plane
    '''

    stat = dict()

    # - 1 - extract Parameters of Acquisition System:
    res_z = parameters['res_z']
    res_xy = parameters['res_xy']
    resolution_factor = res_z / res_xy

    #       dimension of analysis block (parallelogram)
    num_of_slices_P = parameters['num_of_slices_P']
    block_side = row_P = col_P = int(num_of_slices_P * resolution_factor)
    shape_P = np.array((row_P, col_P, num_of_slices_P)).astype(np.uint16)

    #       Pixel size
    pixel_size_F_xy = 1 / (block_side * res_xy)
    pixel_size_F_z = 1 / (num_of_slices_P * res_z)

    # - 2 - extract peaks coordinates from R
    freq_map = R['freq_info']
    peaks_in_pixel = list(R[freq_map]['peak_coord'])

    #       create list of tuple(y,x,z) from list of array(array(r,c,z))
    # nb_3 - traslo le coordinate per avere l'origine degli assi al centro del cubo
    # nb_4 - gli assi dello spettro originale sono [0 -> 31] --> [-16,15] --> allora:
    #    --> aggiungo 0.5 per centrare in [-15.5 , 15.5]
    # value_in_new_axis = (value - trasl) * pixel_size
    peaks_list = [((coord[0][0] - (shape_P[0] / 2) + 0.5) * pixel_size_F_xy,
                   (coord[0][1] - (shape_P[1] / 2) + 0.5) * pixel_size_F_xy,
                   (coord[0][2] - (shape_P[2] / 2) + 0.5) * pixel_size_F_z) for coord in peaks_in_pixel]

    #       move points in Y > 0 subspace
    points_subspace = mirror_in_subspace(peaks_list, subspace=1)

    # - 3 - calculate statistics
    peaks_arr = np.array(points_subspace)

    #       Directionality and modules of vectors
    modules = np.sqrt(peaks_arr[:, 0] ** 2 + peaks_arr[:, 1] ** 2 + peaks_arr[:, 2] ** 2)
    y_norms = peaks_arr[:, 0] / modules  # NB: in Image System, y axis is row axis (0th axis)

    mean_modules = np.mean(modules)
    y_mean = np.mean(y_norms)

    stat['Alignment'] = y_mean
    stat['Mean Module'] = mean_modules  # [um^(-1)]
    stat['Mean Period'] = 1 / mean_modules  # [um]

    #       Standard Deviation and Disorder
    dev = np.zeros(3)
    for coord in range(0, 3):
        dev[coord] = np.std(peaks_arr[:, coord])

    #       variance on XZ plane -> (2th and 3th axis in Image System)
    # areas_ratio : ratio between Ellipse ( with xz axes = std.dev on x and z)
    #               and Circle with radius = mean module
    areas_ratio = (dev[1] * dev[2]) / (mean_modules**2)
    std_dev_norm = np.sqrt(dev[1]**2 + dev[2]**2) / mean_modules
    stat['area_ratio'] = areas_ratio
    stat['sum_std'] = std_dev_norm

    return stat


def mirror_in_subspace(points, subspace):
    # points: list of tuple(r,c,z), every tuple is a 3d point
    # subspace : {integer} - [accepeted values: 0,1,2]
    # - this refer to axes that define selected subspace:
    # - - subspace == 0 -> all points are moved to X>0 subspace
    # - - subspace == 1 -> all points are moved to Y>0 subspace
    # - - subspace == 2 -> all points are moved to Z>0 subspace

    if type(points) is not list:
        points = list(points)

    if subspace is not None and subspace in [0, 1, 2]:
        # in scatter plot Y and X are inverted in Image Standard System
        if subspace == 0:
            mirror_ax = 1
        elif subspace == 1:
            mirror_ax = 0
        else:
            mirror_ax = 2

        # move points in selected subspace
        points_subspace = []  # list of new coordinates
        for p in points:
            # check values on mirror axis:
            if p[mirror_ax] < 0:
                points_subspace.append((-p[0], -p[1], -p[2]))
            else:
                points_subspace.append((p[0], p[1], p[2]))
        return points_subspace

    else:
        # do anything
        raise ValueError(' subspace is None or not in [0, 1, 2]')


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
                   'local_disorder_xy_side',
                   'local_disorder_z_side',
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


def create_coord_by_iter(r, c, z, shape_P, _z_forced=False):
    # create init coordinate for parallelepiped
    
    row = r * shape_P[0]
    col = c * shape_P[1]
    
    if _z_forced:
        zeta = z
    else:
        zeta = z * shape_P[2]
        
    return (row, col, zeta)


def create_R(shape_V, shape_P):
    '''
    Define Results Matrix - 'R'

    :param shape_V: shape of entire Volume of data
    :param shape_P: shape of parallelepipeds extracted for orientation analysis
    :return: empty Result matrix 'R'
    '''

    if any(shape_V > shape_P):
        # shape
        shape_R = np.ceil(shape_V / shape_P).astype(np.int)

        # define empty Results matrix
        total_num_of_cells = np.prod(shape_R)
        R = np.zeros(
            total_num_of_cells,
            dtype=[('id_block', np.int64),  # unique identifier of block
                   ('cell_info', bool),  # 1 if block is analyzed, 0 if it is rejected by cell_threshold
                   ('freq_info', bool),  # 1 if block contains freq. information, 0 otherwise
                   ('cell_ratio', np.float16),  # ratio between cell voxel and all voxel of block
                   ('psd_ratio', np.float32),  # ratio between sum of psd and su of filtered psd
                   ('peak_ratio', np.float16),  # ratio between value of frequency peak and spectrum integral
                   ('local_disorder', np.float16),  # std. dev. of orientation components of neighbour blocks
                   ('init_coord', np.int32, (1, 3)),  # absolute coord of voxel block[0,0,0] in Volume
                   ('peak_coord', np.float32, (1, 3)),  # relative coord to start of the block
                   ('quiver_comp', np.float32, (1, 3)),  # x,y,z component of quiver (peak coord by center of block)
                   ('orientation', np.float32, (1, 3)),  # (rho, phi, theta)
                   ]
        ).reshape(shape_R)  # 3D matrix

        # initialize mask of info to False
        R[:, :, :]['cell_info'] = False
        R[:, :, :]['freq_info'] = False

        print(' - Dimension of Result Matrix:', R.shape)
        return R, shape_R
    else:
        raise ValueError(' Data array dimension is smaller than dimension of one parallelepiped. \n'
                         ' Ensure which data is loaded or modify analysis parameter')


def parameter_normalizer(R, par):
    ''' Normalizes 'par' parameter of R between 0 and 1'''
    maxv = np.max(R[par][np.where(R[par] > 0)])
    minv = np.min(R[par][np.where(R[par] > 0)])

    for z in range(R.shape[2]):
        for r in range(R.shape[0]):
            for c in range(R.shape[1]):
                if R[r, c, z]['cell_info'] == True:
                    R[r, c, z][par] = (R[r, c, z][par] - minv) / (maxv - minv)
    return R


def remove_outlier(R, parameters, par):
    # secondo filtraggio dei quiver per togliere outlier

    k = parameters['k_hyperbole_psd_ratio']
    y0 = parameters['y0_hyperbole_psd_ratio']
    psd0 = parameters['psd0_hyperbole_psd_ratio']
    t_hyp = parameters['threshold_on_hyperbole']

    with_outlier = np.count_nonzero(R['freq_info'])

    for z in range(R.shape[2]):
        for r in range(R.shape[0]):
            for c in range(R.shape[1]):

                if R[r, c, z]['freq_info'] == True:
                    y_comp = np.abs(R[r, c, z]['quiver_comp'][0][0])

                    # if y_comp is under t_hyp, check if psd is under t_hyp constant
                    if y_comp < t_hyp:
                        if R[r, c, z][par] < t_hyp:
                            R[r, c, z]['freq_info'] = False

                    # else, check if psd is under the hyperbole
                    elif R[r, c, z][par] < ((k / (y_comp + y0)) + psd0):
                        R[r, c, z]['freq_info'] = False

    without_outlier = np.count_nonzero(R['freq_info'])
    return R, with_outlier, without_outlier


def threshold_par(R, parameters, par):
    # - 1 - scarto i blocchi che non superano il test del psd_ratio
    thresh_psd = parameters['psd0_hyperbole_psd_ratio']
    before = np.count_nonzero(R['freq_info'])

    for z in range(R.shape[2]):
        for r in range(R.shape[0]):
            for c in range(R.shape[1]):
                if R[r, c, z][par] < thresh_psd:
                    R[r, c, z]['freq_info'] = False

    after = np.count_nonzero(R['freq_info'])
    return R, before, after


def block_analysis(parall, shape_P, parameters, block_side, mask, verbose, lines):
    # where store results
    results = {}

    # boolean info on analysis results
    there_is_cell = False
    there_is_freq = False

    # check if this contains cell
    total_voxel_P = np.prod(shape_P)
    cell_ratio = np.count_nonzero(parall) / total_voxel_P

    if cell_ratio > parameters['threshold_on_cell_ratio']:
        # Orientation Analysis in this data block
        there_is_cell = True

        # save in R
        results['cell_ratio'] = cell_ratio
        if verbose: lines.append('   cell_ratio :   {}'.format(cell_ratio))

        # 3D FFT
        # slices_pad: number of zeros slices added on top and on bottom of spectrumfft_3d
        spec_cube, psd, psd_cube, slices_pad = fft_3d_to_cube(parall,
                                                              int(parameters['num_of_slices_P']),
                                                              block_side)

        # Filtering of spatial frequency with spherical shell mask on spectrum
        spec_cube_filt, psd_cube_filt = filtering_3D(spec_cube, mask)

        # Blurring in frequency space with 3x3x3 Gaussian filter
        # This perform a spheical smoothing in 3d real space
        # (enhance center of block and smooth border)
        psd_cube_blur = gaussian_filter(psd_cube, sigma=parameters['sigma'])
        psd_cube_filt_blur = gaussian_filter(psd_cube_filt, sigma=parameters['sigma'])

        # estimate psd_ratio (TEST)
        results['psd_ratio'] = 10 * np.sum(psd_cube_filt_blur) / np.sum(psd_cube_blur)

        # Find maximum
        peak_coords, peak_ratio = find_peak_in_psd(psd_cube_filt_blur)

        # add peak_ratio to R
        results['peak_ratio'] = peak_ratio
        if verbose: lines.append('   peak_ratio :   {}'.format(peak_ratio))

        # check if there is frequency information in this block
        if peak_ratio > parameters['threshold_on_peak_ratio']:
            there_is_freq = True

            center_of_cube = tuple((np.array(psd_cube_filt_blur.shape) / 2).astype(np.int32))

            # more precision estimation of centroid of spectrum peak
            centroid_coords = find_centroid_of_peak(psd_cube_filt_blur, peak_coords)

            # Spherical coordinates (in degree)
            (rho, phi, theta) = spherical_coord(centroid_coords, center_of_cube)

            # Add info to results
            results['peak_coord'] = (centroid_coords[0],
                                     centroid_coords[1],
                                     centroid_coords[2] - slices_pad)  # real peak position in parallelogram system

            results['orientation'] = (rho, phi, theta)
            results['quiver_comp'] = tuple([np.subtract(c, p) for c, p in zip(centroid_coords, center_of_cube)])

            if verbose: lines.append('   orientation :   {}'.format(results['orientation']))

        else:
            if verbose: lines.append('Block rejected ( no info in freq )')
    else:
        if verbose: lines.append('Block rejected ( no cell )')

    return there_is_cell, there_is_freq, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analysis of orientation in 3d space of selected stack')
    parser.add_argument('-sf', '--source-folder', nargs='+', help='Images to load', required=False)
    main(parser)

