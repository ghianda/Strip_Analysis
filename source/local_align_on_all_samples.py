import argparse
import os
import numpy as np

from custom_tool_kit import printblue, printgreen, search_value_in_txt
from local_alignment_by_R import estimate_local_alignment

def compile_paths(datapath, R_filename=None, param_name='parameters.txt', _nor=False):

    # R filenames
    if R_filename is None:
        R_filename = 'R_gDis4x4x4.npy' if not _nor else 'R_gDis4x4x4_nor.npy'

    # samples paths
    samples_paths = sorted([os.path.join(datapath, d) for d in os.listdir(datapath)
                            if os.path.isdir(os.path.join(datapath, d))])
    # samples names (ID)
    samples_names = [os.path.basename(sp) for sp in samples_paths]

    # R numpy files filepaths
    R_paths = [os.path.join(spath, R_filename) for spath in samples_paths]

    # parameters txt file paths
    param_paths = [os.path.join(spath, param_name) for spath in samples_paths]

    return samples_paths, samples_names, R_paths, param_paths


def extract_parameters(filename, _display=True):
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

    if _display: print('\n ***  Parameters : \n')
    # create dictionary of parameters
    parameters = {}
    for i, p_name in enumerate(param_names):
        parameters[p_name] = float(param_values[i])
        if _display:
            print(' - {} : {}'.format(p_name, param_values[i]))
            print('\n')
    return parameters


def main(parser):

    # parse arguments
    args = parser.parse_args()
    datapath, grane, neighbours_lim, _nor = args.source_folder, int(args.grane), int(args.neighbours_lim), args.nor

    # check neighbours_lim value (default: zero)
    if neighbours_lim == 0:
        default_lim = {3: 14, 4: 30, 5: 60}  # dict of default values {grane: neighbours_lim}
        neighbours_lim = default_lim[grane]

    printgreen('\n******* Local Alignment Estimation Script on All Samples ********')
    printblue('Basepath      : ', end=''), print(datapath)
    printblue('Grane to use  : ', end=''), print(grane)
    printblue('Neighbours Lim: ', end=''), print(neighbours_lim)
    printblue('No Outlier Rem: ', end=''), print(_nor)

    # generate list of paths of directories and files, and the dataframe path
    samples_paths, samples_names, R_paths, param_paths = compile_paths(datapath, _nor=_nor)

    # print all paths generated
    for (listname, currentlist) in zip(['Sample directories', 'R files', 'Parameters files'],
                                       [samples_paths, R_paths, param_paths]):

        printblue('Selected {} paths:'.format(listname))
        for (i, (sname, currentpath)) in enumerate(zip(samples_names, currentlist)):
            print('{0} - {1} - {2}'.format(i + 1, sname, currentpath))

    ''' START ITERATIONS OVER SAMPLES'''
    printgreen('Start to process each sample...')

    for (fldrpath, sname, Rpath, ppath) in zip(samples_paths, samples_names, R_paths, param_paths):
        printblue('\nProcessing sample {} '.format(sname), end='')

        # load R and parameters
        R = np.load(Rpath)
        R_prefix = os.path.basename(Rpath).split('.')[0]
        param = extract_parameters(ppath, _display=False)
        print('from {} with shape: '.format(R_prefix), R.shape)

        # Estimate local Alignment (matrix_of_alignment) and save it in a numpy file
        R, matrix_of_align_perc, shape_G, isolated_value = estimate_local_alignment(R=R,
                                                                                    parameters=param,
                                                                                    grane=grane,
                                                                                    neighbours_lim=neighbours_lim)

        print('- Local Alignment estimated from R Matrix')
        print(' -- grane (r, c, z) used: ({}, {}, {})'.format(shape_G[0], shape_G[1], shape_G[2]))
        print(' -- isolated values set at {}'.format(-1))

        # extract valid_values
        vv = matrix_of_align_perc[matrix_of_align_perc != -1]
        print(' -- Finded {0} valid values, with average: {1:0.1f}% +- {2:0.1f}%'.format(vv.shape[0],
                                                                                         np.mean(vv),
                                                                                         np.std(vv)))

        # save disarray matrix in a numpy file
        mtrx_fname = 'Alignment_matrix_g{}x{}x{}_of_{}.npy'.format(shape_G[0], shape_G[1], shape_G[2], R_prefix)
        np.save(file=os.path.join(fldrpath, mtrx_fname), arr=matrix_of_align_perc)
        print('- Alignment matrix saved as {} with shape: '.format(mtrx_fname), matrix_of_align_perc.shape)
        print('in: ', fldrpath)

    printgreen('****************** Finish Alignment Estimation Script  ******************')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimate Local Alignment of all samples present in given path, '
                                                 'loading each Orientation Matrix (R_[...].npy) '
                                                 'and saving results in a numpy matrix (matrix_of_align_[...].npy.'
                                                 'Local Alignment is evaluated with a space resolution passed in input.')

    parser.add_argument('-sf', '--source-folder', help='basepath of samples data', required=True)

    parser.add_argument('-g', '--grane', help='Space Resoluton (Grane), in terms of number of vectors.',
                        default=4, required=True)

    parser.add_argument('-nl', '--neighbours-lim', default=0, required=False,
                        help='Minimum number of local vectors for evaluation of Alignment.'
                             'Deafult values depend from grane:\n'
                             '- g=3 -> lim = 14 on 27\n'
                             '- g=4 -> lim = 30 on 64\n'
                             '- g=5 -> lim = 60 on 125')

    parser.add_argument('-nor', action='store_true', default=False, dest='nor',
                        help='Add \'-nor\' if you want to load R matrices obtained without outlier remotion. '
                             'Default: False -> Data with Outlier remotion are collected.')
    main(parser)
