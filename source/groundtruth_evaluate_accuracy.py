import os
import argparse

import ast
import pandas as pd
import numpy as np
import tifffile as tiff

pd.set_option('display.max_columns', None)

from custom_tool_kit import printblue, printrose, printbold, printgreen, create_rotation_matrix, yxz_to_polar_coordinates, create_directories, apply_3d_rotations, sigma_from_FWHM, blur_on_z_axis
from gt_constants import settings as gts
from groundtruth_generate_candidates import extract_parameters, extract_orientation_vector


class ERR_FLAG:
    APPLIED = 'applied'
    V0 = 'v0'
    VR = 'vr'
    VEST = 'v_est'
    POLAR_V0 = 'polar_v0'
    POLAR_VR = 'polar_vr'
    POLAR_VEST = 'polar_v_est'
    ERROR = 'error'


def load_candidate_df_from_excel(path, fname, _only_valid=True):
    """
    Load the dataframe of candidates from the excel file in path -> fname.
    Use converters to load columns with the right type
    (ex: str -> tuple for coordinates and vectors

    :param path: path of excel file to load
    :param fname: filename of the excel file to load
    :return: pandas dataframe of candidates
    """

    # create complete filepath
    dataframepath = os.path.join(path, fname)

    # read excel file
    dataframe = pd.read_excel(dataframepath, engine='openpyxl', index_col=0)

    # convert strings to tuples
    cols_with_tuples = ['R_shape', 'coords', '(theta, phi) orig',
                       '(theta, phi) align', 'Y distance', 'v_align']
    dataframe = parse_tuple_columns(dataframe, columns=cols_with_tuples)

    if _only_valid:
        # select only rows of valid candidates
        dataframe = dataframe[dataframe['Aligned'] == 1]

    # confirm loading success
    printblue('Dataframe loaded from: ', end=''), print(dataframepath)
    return dataframe


def parse_tuple_columns(df, columns=list()):
    """
    Parse string into tuple in the dataframe 'df' only for the columns in 'columns'.
    String are parsed in tuple using 'parse_string_in_tuple'
    NaN are trasformed in '()'
    :param df:
    :param columns:
    :return: parsed dataframe
    """

    for col in columns:
        # parse elements from string to tuple
        df[col] = df[col].apply(lambda x: parse_string_in_tuple(x))

    return df


def parse_string_in_tuple(s, _print_excep=False):
    """
    Parse the string s into a tuple.
    NaN are arsed in ()
    """
    try:
        return ast.literal_eval(str(s))
    except Exception as e:
        if _print_excep:
            print(e)
        return ()


def create_error_mtrx(shape_E):

    # define empty Results matrix
    total_num_of_cells = np.prod(shape_E)

    error_mtrx = np.zeros(
        total_num_of_cells,
        dtype=[('applied', np.float32, (2)),  # applied rotation (theta, phi)
               ('v0', np.float32, (3)),  # default orientation (no rot)
               ('vr', np.float32, (3)),  # v0 rotated
               ('v_est', np.float32, (3)),  # orientation estimated by the block analysis
               ('polar_v0', np.float32, (3)),  # polar coordinates of v0 (theta, phi)
               ('polar_vr', np.float32, (3)),  # polar coordinates of vr (theta, phi)
               ('polar_v_est', np.float32, (3)),  # polar coordinates of v_est0 (theta, phi)
               ('error', np.float32),  # module of the error vector (v_est - v_r)
               ]
    ).reshape(shape_E)  # 3D matrix

    return error_mtrx


def orientation_accuracy(filepath, parpath, theta_list, phi_list, mode='wrap', _smooth_z=True, _save_tiffs=False,
                         _save_plot=False,
                         _display_plot=False, plot_path=None, tiff_path=None):
    if _save_plot:
        print('- Vectors plots will be saved in:')
        print(plot_path)

    # = 0 = create error results matrix (theta_angles [raw] x phi_angles [column])
    shape_E = np.array([len(theta_list), len(phi_list)])
    error_mtrx = create_error_mtrx(shape_E=shape_E)

    # = 1 = Analize original block before rotations ------------------------------

    # load tif and move axis to YXZ ref
    block = np.moveaxis(tiff.imread(filepath), 0, -1)  # (z, y, x) -> (r, c, z) = (y, x, z)
    print('\n- Tiff loaded.')

    # reads parameters (strip)
    parameters = extract_parameters(parpath)

    # orientation analysis
    there_is_freq, v0 = extract_orientation_vector(vol=block, param=parameters)
    rho0, theta0, phi0 = yxz_to_polar_coordinates(v0)
    if there_is_freq:
        print('- Original orientation extracted.')
    else:
        printrose('In the tissue there is no frequency information')
        v0 = None

    # save original orientation in the whole error_matrix
    error_mtrx[:, :][ERR_FLAG.V0] = v0
    error_mtrx[:, :][ERR_FLAG.POLAR_V0] = np.array([rho0, theta0, phi0])

    # todo remove
    # if _display_plot or _save_plot:
    #     title = 'V0 Estimated by original block'
    #     fig_filename = 'V0.pdf'
    #     save_quiver_plot(v0, theta0, phi0, title=title, _display=_display_plot, _save=_save_plot,
    #                      path=plot_path, fig_filename=fig_filename)

    # = 2 = Apply rotations --------------------------------------------------------

    for (r, theta_deg) in enumerate(theta_list):
        for (c, phi_deg) in enumerate(phi_list):

            # if current rotation is not (theta=0, phi=0)
            #             if (theta_deg, phi_deg) != (0.0, 0.0):
            if True:
                print('.', end='')

                error_mtrx[r, c][ERR_FLAG.APPLIED] = np.array([theta_deg, phi_deg])

                # create rotation matrices
                rot_theta = create_rotation_matrix(angle_in_deg=theta_deg, axis=1)
                rot_phi = create_rotation_matrix(angle_in_deg=phi_deg, axis=2)
                rot_mtrx = rot_phi * rot_theta

                # apply rotation to v0 and save new polar coordinates:
                vr = rot_mtrx.apply(v0)
                rho_r, theta_r, phi_r = yxz_to_polar_coordinates(vr)
                error_mtrx[r, c][ERR_FLAG.VR] = vr
                error_mtrx[r, c][ERR_FLAG.POLAR_VR] = np.array([rho_r, theta_r, phi_r])

                # plot vector rotated
                #                 if _display_plot or _save_plot:
                #                     title = 'VR: V0 rotated by matrix [φ * θ] with (θ = {0:0.1f}, φ = {1:0.1f})'.format(theta_deg, phi_deg)
                #                     fig_filename = 'θ{0:0.1f}_φ{1:0.1f}_VR.pdf'.format(theta_deg, phi_deg)
                #                     save_quiver_plot(vr, theta_r, phi_r, title=title, _display=_display_plot, _save=_save_plot,
                #                                      path=plot_path, fig_filename=fig_filename)

                # create new rotated volume
                block_rotated = apply_3d_rotations(vol=block, theta=theta_deg, phi=phi_deg,
                                                   res_xy=parameters['res_xy'], res_z=parameters['res_z'],
                                                   mode=mode)

                if _smooth_z:
                    # smooth on z axis to simulate FWHMz of acquisition system
                    sigma_z_px = sigma_from_FWHM(FWHM_um=3.1, px_size=parameters['res_z'])
                    block_ready = blur_on_z_axis(vol=block_rotated, sigma_px=sigma_z_px)
                else:
                    block_ready = block_rotated.copy()

                if _save_tiffs:
                    filename = 'θ{0:0.1f}_φ{1:0.1f}.tif'.format(theta_deg, phi_deg)
                    outfilepath = os.path.join(tiff_path, filename)
                    # save tiff with right swapped axis (yxz -> zyx)
                    tiff.imsave(outfilepath, np.moveaxis(block_ready, -1, 0))

                # estimate orientation of rotated block
                there_is_freq_in_ve, ve = extract_orientation_vector(vol=block_ready, param=parameters)

                if not there_is_freq_in_ve:
                    printrose('There is no freq. info in the rotated volume.')
                    ve = np.array([0, 0, 0])

                rho_e, theta_e, phi_e = yxz_to_polar_coordinates(ve)
                error_mtrx[r, c][ERR_FLAG.VEST] = ve
                error_mtrx[r, c][ERR_FLAG.POLAR_VEST] = np.array([rho_e, theta_e, phi_e])

                # evaluate module of error vector: |V0 - VR|
                error_mtrx[r, c][ERR_FLAG.ERROR] = np.linalg.norm(ve - vr)

    return error_mtrx, parameters



def main(parser):
    """

    :param parser:
    :return:
    """

    '''------------------------------------ Preliminary operations ------------------------------------'''
    # parse arguments
    args            = parser.parse_args()
    datapath        = args.source_folder
    dataframe_fname = args.dataframe
    param_fname     = args.parameters_fname

    printgreen('\n******* Groundtruth Candidates Generation Script ********')
    printblue('Basepath: ', end=''), print(datapath)

    # generate path of results (error matrices)
    errmtrx_basepath = os.path.join(datapath, 'error_matrices')
    create_directories([errmtrx_basepath])

    # todo
    if gts['_save_heatmap']:
        # generate path of 'single' heatmaps (from the error matrix of each candidate)
        pass

    # load dataframe
    candidates_df = load_candidate_df_from_excel(datapath, dataframe_fname, _only_valid=True)

    printblue('Selected valid candidates:')
    print(candidates_df)

    '''------------------------------------ Accuracy evaluation ------------------------------------'''
    printgreen('Start to evaluate accuracy over candidates.')
    printblue('Settings:')
    for k in gts.keys():
        printbold('{0:20s}'.format(k), end=': ')
        print(gts[k])
    print()

    # todo capire se servono
    # plots and tiffs folder paths
    plot_path = os.path.join(datapath, gts['plt_folder'])
    tiff_path = os.path.join(datapath, gts['tiff_folder'])

    for (index, row) in candidates_df.iterrows():
        printblue('Candidate Index: {0:3} - from {1}'.format(index, row['Sample_id']))

        # todo - per ogni candidato ------------------------------------:
        # - eseguire i test di accuracy (vedi JupyterProjects/Strip/accuracy_orientation/single_block/single_block_analysis_accuracy_STRIP_heatmaps_articolo_BOCCHI.ipynb
        # - ricordarsi che gli errori siano relativi (non valore assoluto)
        # - salvare per ogni candidato la sua matrice di errore
        # dopo di che, su un notebook, colleziono tutte le matrici in un numpy array 3d, e genero grafici a gogo
        # todo ---------------------------------------

        # accuracy analysis
        error_mtrx, parameters = orientation_accuracy(filepath=row['inner_path'],
                                                      parpath=os.path.join(datapath, param_fname),
                                                      theta_list=gts['theta_list'],
                                                      phi_list=gts['phi_list'],
                                                      mode=gts['mode'],
                                                      _smooth_z=gts['_smooth_z'],
                                                      _save_tiffs=gts['_save_tiffs'],
                                                      _save_plot=gts['_save_plot'],
                                                      _display_plot=gts['_display_plot'],
                                                      plot_path=plot_path,
                                                      tiff_path=tiff_path)

        # save current error matrix in a numpy file
        errmtrx_fname = 'error_mtrx_cand{}_s{}.npy'.format(index, row['Sample_id'])
        np.save(os.path.join(errmtrx_basepath, errmtrx_fname), error_mtrx)

        # todo
        if gts['_save_heatmap']:
            # blocco codice jupyter che genera le due heatmap del campione corrente
            pass




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load dataframe and subvolumes and perform the analysis of accuracy'
                                                 ' on orientation estimation.')

    parser.add_argument('-sf', '--source-folder',
                        help='basepath of data to load (subvolumes and dataframe)',
                        required=True)

    parser.add_argument('-d', '--dataframe',
                        help='filename of dataframe (excel file)',
                        required=False,
                        default='groundtruth.xlsx')

    parser.add_argument('-p', '--parameters-fname',
                        help='filename of the parameters.txt file to load',
                        required=False,
                        default='parameters.txt')

    main(parser)