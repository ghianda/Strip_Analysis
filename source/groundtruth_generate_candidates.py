import numpy as np
from numpy.random import randint
import os
import argparse
import pandas as pd
import tifffile as tiff
import shutil

from custom_tool_kit import printblue, printrose, printbold, printgreen, search_value_in_txt, create_directories, \
    yxz_to_polar_coordinates, apply_3d_rotations, sigma_from_FWHM, blur_on_z_axis

from GAMMA_orientation_analysis import block_analysis
from custom_freq_analysis import create_3D_filter


def from_virtual_to_real_around(candidate, volume, shape_P, _verb=False):
    """
    Select block pointed by 'candidate_coords' inside R. Get its real_coordinates in the 'volume' (init_coord)
    Extract a 3x3x3 "block" sub-volumes from 'volume' centered in the candidate real coordinates.
    S.R. YXZ
    :param candidate_coords: candidate coordinate in the R s.r.
    :param volume: real tiff volume (as numpy nd-array) of the entire sample
    :param shape_P: shape of the elementary block
    :return: sub-volumes extracted from vol as a numpy nd-array

    """

    # extract real coordinates in the volume ref system
    init_coord = candidate['init_coord'].squeeze()

    # evaluate start coordinates of the external block
    init_ext = init_coord - shape_P

    # evaluate shape of the external volume
    shape_ext = np.array(shape_P) * 3

    if _verb:
        for (label, info) in zip(['shape_P', 'shape_ext', 'init_coord', 'init_ext'],
                                 [shape_P, shape_ext, init_coord, init_ext], ):
            printgreen(label, end=' - '), print(info, end=' - ')
        print()

    around = extract_subvolume(vol=volume, start_sub=init_ext, shape_sub=shape_ext)
    return around


def from_virtual_to_real_inner(candidate, volume, shape_P):
    """
    Extract a block with shape:'shape_P' pointed by 'candidate_coords' inside R, by its real_coordinates in the 'volume' (init_coord).
        S.R. YXZ
    :param candidate_coords: candidate coordinate in the R s.r.
    :param volume: real tiff volume (as numpy nd-array) of the entire sample
    :param shape_P: shape of the elementary block
    :return: sub-volumes extracted from vol as a numpy nd-array
    """
    # extract real coordinates in the volume ref system
    init_coord = candidate['init_coord'].squeeze()

    # extract subvolume
    inner = extract_subvolume(vol=volume, start_sub=init_coord, shape_sub=shape_P)
    return inner


def extract_subvolume(vol, start_sub, shape_sub, _verb=False):
    """
    Extract from vol a subvolumes with shape 'shape_sub' located at 'start_sub'
    S.R. YXZ
    """
    try:
        subvol = np.copy(vol[start_sub[0] : start_sub[0] + shape_sub[0],
                             start_sub[1] : start_sub[1] + shape_sub[1],
                             start_sub[2] : start_sub[2] + shape_sub[2]])
    except:
        print('vol[slice_coord] -> there is some problem')
        return None

    return subvol


def extract_inner_from_around(around, shape_P):
    """
    Extract a central subvolume from 'around'.
    Subvlums has shape = shape_P
    :param around: big volume
    :param shape_P: shape of the subvolume to extract
    :return: inner volume (a subvolumes with shape = shape_P) in the (3D) center of 'around' volume
    """

    if around is not None and all(around.shape > shape_P):

        # inner = np.copy(around[shape_P[0] : 2 * shape_P[0],
        #                        shape_P[1] : 2 * shape_P[1],
        #                        shape_P[2] : 2 * shape_P[2]])
        #
        inner = extract_subvolume(vol=around, start_sub=shape_P, shape_sub=shape_P, _verb=False)
        return inner
    else:
        printrose('There is a problem to extract inner from the around:')
        printrose('around.shape: '), printrose(str(around.shape))
        printrose('requested inner shape: '), printrose(shape_P)
        return None


def extract_orientation_vector(vol, param, _verb=False):
    """
    Evaluate 3D orientation of sarcomeres periodicity inside 'vol'.
    Return boolean variable (if block contains frequency informatons),
    and the normalized orientation vector as (y,x,z).
    """

    # spatial analysis resolution
    num_of_slices_P = param['num_of_slices_P']
    resolution_factor = param['res_z'] / param['res_xy']
    block_side = int(num_of_slices_P * resolution_factor)
    shape_P = np.array((block_side, block_side, num_of_slices_P)).astype(np.int32)

    # create frequency filter
    mask = create_3D_filter(block_side, param['res_xy'], param['sarc_length'])

    # fft orientation analysis
    _, there_is_freq, results = block_analysis(vol,
                                               shape_P,
                                               param,
                                               block_side,
                                               mask,
                                               verbose=_verb,
                                               lines=[])

    if there_is_freq:

        # extract orientation from results
        v = np.array(results['quiver_comp'])

        # normalize vector
        v = v / np.linalg.norm(v)

        # turn with y > 0
        if v[0] < 0:
            v = -v

        return there_is_freq, v
    else:
        return there_is_freq, None


def get_shape_P(param):
    slices = param['num_of_slices_P']
    resolution_factor = param['res_z'] / param['res_xy']
    shape_P = np.array((int(slices * resolution_factor),
                        int(slices * resolution_factor),
                        slices)).astype(np.int32)
    return shape_P


def extract_parameters(filename, _display=True):
    ''' read parameters values in filename.txt
    and save it in a dictionary'''

    param_names = ['num_of_slices_P',
                   'res_xy', 'res_z',
                   'sarc_length',
                   'threshold_on_cell_ratio',
                   'sigma',
                   'threshold_on_peak_ratio']

    # read values in txt
    param_values = search_value_in_txt(filename, param_names)

    # create dictionary of parameters
    parameters = {}
    for i, p_name in enumerate(param_names):
        parameters[p_name] = float(param_values[i])
        if _display: print(' - {} : {}'.format(p_name, param_values[i]))
    return parameters


def check_cell_content(matrix):
    """
    Check if 'cell_info' parameter is True in all 'matrix' elements
    :param matrix: input matrix containing elements to check
    :return boolean
    """
    if matrix is not None:
        return matrix['cell_info'].all()
    else:
        return False


def extract_virtual_around(coord, R, _verb=False):
    """
    Extract from 'R' a 3x3x3 matrix around 'coord' called 'around'
    if candidate (coord) is on boundaries of R, return None

    :param coord: tuple of coordinate of the candidate inside R
    :param R: orientation matrix
    :param _verb: if True, function prints info to console
    :return: 3x3x3 matrix of elements around central element identified by 'coord'
    """
    if _verb: print('Checking: ', coord, ' on R matrix with shape: ', R.shape)

    # check boundaries of R
    if all(np.array(coord) > 0) and all(coord < np.array(R.shape)):
        if _verb: print('A: ', all(np.array(coord) > 0))
        if _verb: print('B: ', all(coord < np.array(R.shape)))
        if _verb: print('Ok, no on boundaries.')

        # divide indexes
        (r, c, z) = coord

        # extract 3x3x3 'around'
        around = R[r - 1:r + 2, c - 1:c + 2, z - 1:z + 2]
        if _verb: print('Extracted around with shape: ', around.shape)
        return around

    else:
        if _verb: print('FAIL: candidate on boundaries of R.')
        return None


def find_a_candidate(R, shape_P, _verb=False):
    """
    Find a valid candidate inside R.
    A candidate is valid if it has 'freq_info' = True and
    the 3x3x3 cells around have 'cell_info' == True.

    :param R: orientation matrix
    :param shape_P: dimension of single block in the volume R.S.
    :param _verb: if True, function prints info to console
    :return: a single row dataframe with candidate coordinates and number of attempts to find it.
    """
    # extract blocks with only frequency blocks
    R_freq = R[R['freq_info'] == True]
    freqs = R_freq.shape

    # initialize loop
    finded = False
    attempts = 0
    max_attempts = np.prod(R.shape)

    # try to find a valid candidate
    while (finded is False) and (attempts < max_attempts):
        attempts = attempts + 1

        # generate random index (sure with freq)
        rand_i = randint(0, freqs)
        if _verb:
            print('\n Test n: {} - random index: {} on {}'.format(attempts, rand_i, freqs))

        # extract current candidate
        candidate = R_freq[rand_i]

        # generate virtual and real coordinates
        (rR, cR, zR) = tuple(np.array(candidate['init_coord'] / shape_P).squeeze().astype(np.int64))

        if _verb:
            print('Virtual coordinates (R): ', rR, cR, zR)

        # extract around
        around = extract_virtual_around(coord=(rR, cR, zR), R=R, _verb=False)

        # check around content
        is_full = check_cell_content(around)

        if is_full:
            # create the record
            # old:
            candidate_df = pd.Series([tuple((rR, cR, zR)), attempts], index=['coords', 'attempts'])
            # new:
            # candidate_df = pd.DataFrame({'coord': tuple((rR, cR, zR)),
            #                              'attempts:': attempts})

            # set flag True
            finded = True

    if _verb:
        print('+++++++++++++++++++++++++++++++++')
        print('Solved in {} attempts'.format(attempts))
        print('Valid candidate is: ', ((rR, cR, zR)))

    if finded is True:
        # loop finished successfully
        return candidate_df
    else:
        # loop finish without success
        return pd.DataFrame({'coord': tuple((0, 0, 0)),
                             'attempts': max_attempts})


def find_candidates(R, n_candidates, shape_P, _verb=True):
    """
    Return a dataframe containing coordinates of 'n_candidates' valid elements from R.
    A candidate is valid if it has 'freq_info' = True and
    the 3x3x3 cells around have 'cell_info' == True.

    :param R: input orientation matrix
    :param n_candidates: number of valid elements to individuate
    :param shape_P: dimension of single block in the volume R.S.
    :param _verb: if True, function prints info to console
    :return: dataframe containing for each candidate (row):
    - coords: (r, c, z) - tuple of coordinate in the 'R' matrix reference system;
    - attempts: numer of attempts performed to find it
    """

    # extrat R axis dimension
    rows, cols, zetas = R.shape
    if _verb: print('Start to search candidates inside R matrix with shape: ', rows, cols, zetas)

    # number of total blocks, with cell_info, and freq_info
    blocks = np.prod(R.shape)
    cells  = np.prod(R[R['cell_info'] == True].shape)
    freqs  = np.prod(R[R['freq_info'] == True].shape)

    if _verb: print('- Blocks with cell: {0} on {1} ({2:0.1f}%)'.format(cells, blocks, 100*cells/blocks))
    if _verb: print('- Blocks with freq: {0} on {1} ({2:0.1f}%)'.format(freqs, blocks, 100*freqs/blocks))

    # define dataframe of candidates
    candidates_df = pd.DataFrame(columns=['coords', 'attempts'])

    # define index name
    candidates_df.index.name = 'id_cand'

    # # set columns type (coords is set as string to avoid error when I write a tuple there)
    # candidates_df = candidates_df.astype({'coords': str,
    #                                       'attempts': int})

    # initialize research index
    while len(candidates_df) < n_candidates:

        if _verb: print('\nSearching the {}th candidate...'.format(len(candidates_df) + 1))

        # extract a new candidate
        new_candidate_df = find_a_candidate(R=R, shape_P=shape_P, _verb=False)

        # check if it is already present in candidates_df
        if new_candidate_df.coords in set(candidates_df.coords.unique()):
            # discard current candidate
            if _verb: print(new_candidate_df.coords, ' is already present in the list:')
            pass
        else:
            # add to the candidate list
            candidates_df = candidates_df.append(new_candidate_df, ignore_index=True)
            if _verb: print('Finded', new_candidate_df.coords,
                            'in {} attempts'.format(new_candidate_df.attempts))

    # last check of duplicates
    if all(candidates_df.coords.value_counts() == 1):
        if _verb: print('\nSuccessfully finded {} valid candidates'.format(n_candidates))
        return candidates_df
    else:
        return None


def compile_paths(datapath, R_filename='R_gDis4x4x4.npy', param_name='parameters.txt'):

    # samples paths
    samples_paths = sorted([os.path.join(datapath, d) for d in os.listdir(datapath)
                            if os.path.isdir(os.path.join(datapath, d))])
    # samples names (ID)
    samples_names = [os.path.basename(sp) for sp in samples_paths]

    # R numpy files filepaths
    R_paths = [os.path.join(spath, R_filename) for spath in samples_paths]

    # parameters txt file paths
    param_paths = [os.path.join(spath, param_name) for spath in samples_paths]

    tiff_paths = [os.path.join(spath, tiff) for spath in samples_paths for tiff in os.listdir(spath)
                  if os.path.isfile(os.path.join(spath, tiff)) and tiff.endswith('.tif')]

    return samples_paths, samples_names, R_paths, param_paths, tiff_paths


def compile_subvolumes_paths(datapath, samples_names, _in_ram=False):

    # generate base path
    if not _in_ram:
        subvolumes_basepath = os.path.join(os.path.dirname(datapath), 'subvolumes')
    else:
        subvolumes_basepath = os.path.join('/mnt/ramdisk', 'subvolumes')

    # generate subfolders for each sample
    samples_sub_paths = sorted([os.path.join(subvolumes_basepath, d) for d in samples_names])
    return samples_sub_paths


def generate_dataframe(samples_names, n_cand):

    # generate dataframe
    dataframe = pd.DataFrame()
    dataframe['Sample_id'] = samples_names  # add list of samples names (id)

    # duplicate rows for each block to test (X n_cand)
    cols = dataframe.columns.tolist()  # columns to be replicated (all)

    # new colomn
    newcol, newcol_values = 'n_cand', list(range(0, n_cand))

    # creo nuovo dataframe di appoggio: {newcol, app}
    temp_df = pd.DataFrame({newcol: newcol_values, 'app': 1})

    # combino con il dataframe esistende agganciando la colonna 'app' e poi rimuovendola
    dataframe = dataframe[cols].assign(temp=1).merge(temp_df, on='app').drop('app', axis=1)
    return dataframe


def save_dataframe(dataframe, dataframepath):

    # save dataframe to an excel file
    dataframe.to_excel(excel_writer=dataframepath)

    # display destination path
    printblue('Dataframe saved in:')
    print(dataframepath)


def copy_paste_file(frompath, topath, _in_ram=False):
    '''
    Copy the file 'frompath' and paste it into 'topath'.
    If _in_ram is TRue, copy the file in /mnt/ramdisk'

    :param frompath: source filepath
    :param topath: dest filepath
    :param _in_ram: boolean
    :return path of file copied
    '''

    if os.path.exists(frompath):
        if _in_ram:
            topath = os.path.join('/mnt/ramdisk', os.path.basename(frompath))
        else:
            if not os.path.exists(topath):
                printrose('ERROR - destination path not exists')
                return None

        # copy file
        shutil.copyfile(frompath, topath)

    else:
        printrose('ERROR - Source file not found')

    return topath



def compile_dataframe_path(fname='noname.xlsx', fldrpath='/mnt/ram/', _in_ram=False):

    # select destination
    if _in_ram:
        dataframepath = os.path.join('/mnt/ramdisk/', fname)
    else:
        dataframepath = os.path.join(os.path.dirname(fldrpath), fname)

    return dataframepath


def clean_folder(folder):

    printblue('Deleting files from: ', end=''), print(folder)

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            # del file
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            # del subfolder
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def main(parser):
    """

    :param parser:
    :return:
    """

    '''------------------------------------ Preliminary operations ------------------------------------'''
    # parse arguments
    args         = parser.parse_args()
    datapath     = args.source_folder
    n_candidates = int(args.number_of_candidates)
    n_test       = int(args.number_of_test)
    _ram         = bool(args.save_in_ram)

    printgreen('\n******* Groundtruth Candidates Generation Script ********')
    printblue('Basepath: ', end=''), print(datapath)
    printblue('Number of candidates for each sample: ', end=''), print(n_candidates)
    printblue('Number of test to perform for each sample: ', end=''), print(n_test)
    printblue('Save output in ram: ', end=''), print(_ram)

    # generate list of paths of directories and files, and the dataframe path
    samples_paths, samples_names, R_paths, param_paths, tiff_paths = compile_paths(datapath)

    # generate path of the output dataframe
    dataframe_path = compile_dataframe_path(fname='groundtruth.xlsx', fldrpath=datapath, _in_ram=_ram)

    # generate paths of results directories, an create directories if they not exists
    samples_sub_paths = compile_subvolumes_paths(datapath, samples_names, _in_ram=_ram)
    create_directories(samples_sub_paths)

    # delete data from ram if present
    if _ram:
        clean_folder(folder='/mnt/ramdisk/')

    # create, print, and save empty dataframe
    dataframe = pd.DataFrame()
    save_dataframe(dataframe, dataframe_path)

    # print all paths generated
    for (listname, currentlist) in zip(['Sample directories', 'R files', 'Parameters files', 'Tiff files', 'Sub Volumes'],
                                       [samples_paths, R_paths, param_paths, tiff_paths, samples_sub_paths]):

        printblue('Selected {} paths:'.format(listname))

        for (i, (sname, currentpath)) in enumerate(zip(samples_names, currentlist)):
            print('{0} - {1} - {2}'.format(i + 1, sname, currentpath))

    # extract parameters from the first sample path (all samples have the same parameters)
    printblue('Loaded parameters from:')
    print(param_paths[0])
    param = extract_parameters(param_paths[0])

    # generate a copy of the parameters file in the output folder
    copied_param_filepath = copy_paste_file(frompath=param_paths[0],
                                            topath=os.path.join(datapath, 'parameters.txt'),
                                            _in_ram=_ram)
    printblue('Parameters file correctly copied into:')
    print(copied_param_filepath)
    print()

    '''-- [1] --------------------------------- Start Collecting Candidates ------------------------------------'''

    printgreen('Start to collect candidates for each sample...')

    # find a list of valid candidates for each sample, and write the list of their coordinates in the dataframe
    # for (sname, Rpath, ppath, tpath) in zip(samples_names, R_paths, param_paths, tiff_paths):
    for (sname, Rpath, ppath, tpath) in zip(samples_names, R_paths, param_paths, tiff_paths):
        print('Extracting candidates for sample {} '.format(sname), end='')

        # load R and parameters
        R = np.load(Rpath)
        print('from R with shape: ', R.shape)

        # evaluate shape_P from parameters:
        shape_P = get_shape_P(param)

        # find candidates
        candidates_df = find_candidates(R=R, n_candidates=n_candidates, shape_P=shape_P, _verb=False)

        # add info to all the rows of the current candidates_df
        candidates_df['Sample_id'] = sname
        candidates_df['R_shape'] = '({}, {}, {})'.format(R.shape[0], R.shape[1], R.shape[2])
        candidates_df['n_cand'] = list(range(0, len(candidates_df)))

        # append candidates of the current sample to the global dataframe
        dataframe = dataframe.append(candidates_df, ignore_index=True)

        # del current candidates
        del candidates_df

    # re-order columns
    dataframe = dataframe[['Sample_id', 'R_shape', 'n_cand', 'coords', 'attempts']]

    # save dataframe in the excel file
    save_dataframe(dataframe, dataframe_path)

    '''-- [2] -----------------[ Start Extracting Volumes and Perform Alignement to Y axis ]------------------------'''

    # # create dictionaries of sample_paths and tiff_paths with sample_name (D1, D2..) as keys
    # d_samples_paths = lists_to_dict(samples_names, samples_paths)
    # d_tiff_paths    = lists_to_dict(samples_names, tiff_paths)
    #
    # printblue('Generated dictionaries of paths:')
    # printdict(d_samples_paths)
    # printdict(d_tiff_paths)

    printgreen('Start to extract sub-volumes around candidates for each sample...')

    # iterate on samples (to load the entire volume tiff only one time)
    for (sname, Rpath, ppath, tiffpath, subfldrpath) in zip(samples_names, R_paths,
                                                            param_paths, tiff_paths, samples_sub_paths):

        printgreen('\nSample: {}'.format(sname))

        # load tiff of current sample
        vol = np.moveaxis(tiff.imread(tiffpath), 0, -1)  # (z, y, x) -> (r, c, z) = (y, x, z)
        print('Volume of {} correctly loaded. Shape: '.format(sname), vol.shape)

        # load R and shape_P
        R, shape_P = np.load(Rpath), get_shape_P(param)

        # extract indexes of candidates of the current sample from the dataframe
        current_indexes = dataframe.index[dataframe['Sample_id'] == sname].tolist()

        # iterate on candidates until system find 'n_test' well aligned candidates to use for the performance analysis
        checked = 0  # number of candidates extracted, ruotated and checked if aligned with Y
        aligned = 0  # number of candidates well aligned with Y
        failed  = 0  # number of candidates discarded

        # iter on the current subset (current sample) of global indexes
        for index in current_indexes:

            # todo [From HERE to THERE] -> may be inside: -----------------------------------------------
            # perform_registration_to_Y_axis(....)

            # select candidate
            row = dataframe.loc[index]

            printblue('Candidate Number: {} - R_coords: ({}, {}, {})'.format(index,
                                                                             row['coords'][0],
                                                                             row['coords'][1],
                                                                             row['coords'][2]))

            # current candidate subvolumes path
            currentsubpath = os.path.join(subfldrpath, 'cand_{}'.format(index))
            create_directories([currentsubpath])

            # extract inner volume
            inner = from_virtual_to_real_inner(R[row['coords']], vol, shape_P)
            # print('Extracted inner volume with shape:', inner.shape)

            # extract around volume
            around = from_virtual_to_real_around(R[row['coords']], vol, shape_P)
            # print('Extracted around volume with shape:', around.shape)

            # todo - poi rimuoverò questi salvataggi temporanei in tiff
            # save inner and around subvolumes as tiff swapping axis in the right way (yxz -> zyx)
            tiff.imsave(os.path.join(currentsubpath, '{}_cand_{}_inner.tiff'.format(sname, index)),
                        np.moveaxis(inner, -1, 0))
            tiff.imsave(os.path.join(currentsubpath, '{}_cand_{}_around.tiff'.format(sname, index)),
                        np.moveaxis(around, -1, 0))

            # analyze orientation of inner block
            _, v0 = extract_orientation_vector(vol=inner, param=param)
            rho0, theta0, phi0 = yxz_to_polar_coordinates(v0)
            print('Extracted original orientation: (rho, theta, phi) = ({0:0.1f}, {1:0.1f}, {2:0.1f})'.format(
                rho0, theta0, phi0))

            # save original orientation in the dataframe
            dataframe.loc[index, '(theta, phi) orig'] = '({0:0.1f}, {1:0.1f})'.format(theta0, phi0)

            # evaluate opposite rotation to apply to the 'around' volume - to align it to the Y axis
            theta_to_apply = 90 - theta0  # theta = 90 when aligned with Y axis (i.e. on the xy plane)
            phi_to_apply   = 0 - phi0  # phi = 0 when aligned with Y axis

            # apply 3d rotation
            around_rotated = apply_3d_rotations(vol=around, theta=theta_to_apply, phi=phi_to_apply,
                                               res_xy=param['res_xy'], res_z=param['res_z'],
                                               mode='wrap')
            # print('Applied alignment to the around volume: ({0:0.1f}, {1:0.1f})'.format(theta_to_apply, phi_to_apply))

            # todo - ma serve?
            # save middle frame of ext with bands superimposed
            fname = '{0}_cand{1}_around_aligned_by_theta{2:0.1f}_phi{3:0.1f}.tiff'.format(
                sname, index, theta_to_apply, phi_to_apply)
            # todo remove and use 'save_middle_frame'
            tiff.imsave(os.path.join(currentsubpath, fname), np.moveaxis(around_rotated, -1, 0))
            # save_middle_frame_as_tiff(vol=around_rotated, filename=fname, path=currentsubpath)

            # smooth on z axis to simulate FWHMz of acquisition system
            sigma_z_px = sigma_from_FWHM(FWHM_um=3.1, px_size=param['res_z'])
            around_rotated = blur_on_z_axis(vol=around_rotated, sigma_px=sigma_z_px)

            # extract inner from the rotated around
            inner_aligned = extract_inner_from_around(around_rotated, shape_P)
            # print('Extracted inner from aligned around -> inner.shape:', inner_aligned.shape)

            # TODO togliere (salvare solo quelli allineati bene
            tiff.imsave(os.path.join(currentsubpath, '{}_cand_{}_inner_aligned.tiff'.format(sname, index)),
                        np.moveaxis(inner_aligned, -1, 0))

            # evaluate orientation of inner_rotated for control (it must be (90, 0) if well aligned)
            there_is_freq, v_align = extract_orientation_vector(vol=inner_aligned, param=param)

            # check if remain frequency information after the 3D rotation
            if there_is_freq:
                dataframe.loc[index, 'Freq after rotation'] = True

                # polar coordinates of the new orientation
                rho_align, theta_align, phi_align = yxz_to_polar_coordinates(v_align)
                dataframe.loc[index, '(theta, phi) align'] = '({0:0.1f}, {1:0.1f})'.format(theta_align, phi_align)
                # print('Orientation of aligned inner: (rho, theta, phi)   = ({0:0.1f}, {1:0.1f}, {2:0.1f})'.format(
                #     rho_align, theta_align, phi_align))

                # check if well aligned with Y axis
                d_theta, d_phi = np.abs(90 - theta_align), np.abs(0 - phi_align)
                print('Absolute distance to the Y axis: (theta, phi) = ', end='')
                printrose('({0:0.2f}, {1:0.2f})'.format(d_theta, d_phi), end='')
                dataframe.loc[index, 'Y distance'] = '({0:0.1f}, {1:0.1f})'.format(d_theta, d_phi)

                if (d_theta + d_phi) < 3:
                    dataframe.loc[index, 'Aligned'] = True

                    # save orientatoin vector in the dataframe with 5 digit decimals
                    dataframe.loc[index, 'v_align'] = '({0:0.5f}, {1:0.5f}, {2:0.5f})'.format(
                        v_align[0], v_align[1], v_align[2])

                    # save well aligned inner for the next step (performance evaluations)
                    inner_path = os.path.join(currentsubpath, '{}_cand_{}_inner_aligned.tiff'.format(sname, index))
                    tiff.imsave(inner_path, np.moveaxis(inner_aligned, -1, 0))

                    # add inner path to the dataframe
                    dataframe.loc[index, 'inner_path'] = inner_path

                    # increment counter
                    aligned = aligned + 1
                    printbold(' -----> saved')
                else:
                    dataframe.loc[index, 'Aligned'] = False
                    printbold(' ----------------> discard')
                    failed = failed + 1

            else:
                row['Freq after rotation'] = False
                row['Aligned'] = False
                print('No frequency information after the 3D rotation. ', end='')
                printbold(' ----------------> discard')
                failed = failed + 1

            # increment counter of checked blocks
            checked = checked + 1

            # check number of ready candidates finded until now
            print('Finded {} block on {} until now.'.format(aligned, n_test))
            if aligned >= n_test:
                break
            # TODO - THERE ------------------------------------------------------------------------------

            # end iteration on current sample

            # write performance on dataframe
            dataframe.loc[current_indexes, 'checked'] = checked
            dataframe.loc[current_indexes, 'failed']  = failed

        printrose('Discarded {} su {}'.format(failed, checked))
        # end iteration on samples

    # save dataframe in the excel file
    save_dataframe(dataframe, dataframe_path)

    '''
    # adesso lavoro sulla cartella 'developing_on_two_samples'.
    funziona - nel senso che lui li allinea davvero (vedi candidato 51 del D2) in:
     /home/francesco/LENS/Strip/data/groundtruth/developing_on_two_samples/temp_subvolumes, è da salvare come esempio
    però la maggio parte 'a occhio' sembrano scazzati, mentre lui li vede allineati.
    -> sbagliamo noi? sbaglia lui? bo.. idea è vedere su 7 campioni e 7 test da trovare, quanti decenti ne fa e far vedere ai prof solo quelli :)
    
    nb adesso salva i subvolumes solo dei candidati elaborati, sia che passino il test che lo boccino (ok) -> così vedo diff
    
    il ciclo itera sugli indici, così posso scrivere nel dataframe in tempo reale all'inidce corrente
    
    2) aprire il file excel in
    /home/francesco/LENS/Strip/data/groundtruth/developing_on_two_samples
    crea nuove colonne
    
    una volta capio se posso andare avanti, devo fare la seconda parte dello script che legge:
    - dataframe -> estraendo solo righe con 'aligned' == 1
    - subvolumes -> pescado i blocchi corrispondenti alla riga (vedi sopra)
    
    e per ogni cubetto  applica performance e collezione gli errori
    per ogni cubetto -> salva le matrici di errori
    poi farò script che carica tutto insieme come dataframe o numpy array (vediamo, forse meglio numpy per il 3d)->
    e poi con jupyter creare le heatmap ecc
    
    io usavo:
    JupyterProjects/Strip/accuracy_orientation/single_block/single_block_analysis_accuracy_STRIP_heatmaps_articolo_BOCCHI.ipynb
    
    -> adesso sviluppo seconda fase in :
    groundtruth_evaluate_accuracy.py 
    '''

    printgreen('****************** Finish Groundtruth Script  ******************')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate subvolumes candidates (random portion of samples aligend with Y axis) '
                                                 'for the analysis of accuracy on orientation estimation.')
    parser.add_argument('-sf', '--source-folder', help='basepath of samples data', required=True)
    parser.add_argument('-nc', '--number-of-candidates',
                        help='For each sample, Number of blocks extracted and prepared (aligned to Y-axis) '
                             'for the performance analysis  (Default: 50)',
                        default=20, required=False)
    parser.add_argument('-nt', '--number-of-test',
                        help='For each sample, number of blocks actually tested (Default: 50)',
                        default=20, required=False)
    parser.add_argument('-ram', '--save-in-ram', default=False, action='store_true',
                        help='if passed, script save subvolumes in /ramdisk, otherwise in the input folder')
    main(parser)