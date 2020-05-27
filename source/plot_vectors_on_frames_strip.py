import argparse

import numpy as np
import os
import time
from PIL import Image
from io import BytesIO

from scipy import ndimage
from zetastitcher import InputFile

from custom_tool_kit import create_coord_by_iter, all_words_in_txt, search_value_in_txt
from custom_image_tool import print_info
from make_data import manage_path_argument, load_tif_data

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

plt.rcParams['figure.figsize']=(14,14)


def normalize(img, max_value=255.0, dtype=None):
    max_v = img.max()
    min_v = img.min()

    if dtype is None:
        dtype = img.dtype

    if max_v != 0:
        if max_v != min_v:
            return (((img - min_v) / (max_v - min_v)) * max_value).astype(dtype)
        else:
            return ((img / max_v) * max_value).astype(dtype)
    else:
        return img.astype(dtype)


def float_to_color(values, color_map=cm.viridis, _print_info=False):
    nz = mcolors.Normalize()
    nz.autoscale(values)
    if _print_info: print_info(values)
    colormap = color_map
    return colormap(nz(values))[:]


def chaos_normalizer(values, isolated_value=-1, assign='max'):
    new_vaues = np.copy(values)

    maxv = np.max(values[values >= 0])
    minv = np.min(values[values >= 0])

    new_vaues[new_vaues == isolated_value] = (maxv if (assign == 'max') else minv)

    return (new_vaues - minv) / (maxv - minv)


# color_map STRINGs
COL_XYANGLE = 0
COL_PARAM = 1
COL_ZETA = 2

# image_format STRINGS
IMG_EPS = 'EPS'
IMG_TIFF = 'TIFF'
IMG_SVG = 'SVG'


class Param:
    ID_BLOCK = 'id_block'  # unique identifier of block
    CELL_INFO = 'cell_info'  # 1 if block is analyzed, 0 if it is rejected by cell_threshold
    FREQ_INFO = 'freq_info'  # 1 if block is analyzed, 0 if it is rejected by freq thershold
    CELL_RATIO = 'cell_ratio'  # ratio between cell voxel and all voxel of block
    PSD_RATIO = 'psd_ratio'  # ratio between filtered and not filtered PSD
    LOCAL_DISORDER = 'local_disorder'
    INIT_COORD = 'init_coord'   # absolute coord of voxel = block[0,0,0] in Volume s.r.
    PEAK_COORD = 'peak_coord'  # coordinate of peak in the block s.r.
    QUIVER_COMP = 'quiver_comp'  # quiver components
    # ORIENTATION = 'orientation'  # (rho, theta, phi)
    # EW = 'ew'   # descending ordered eigenvalues.
    # EV = 'ev'   # column ev[:,i] is the eigenvector of the eigenvalue w[i].
    # STRENGHT = 'strenght'   # parametro forza del gradiente (w1 .=. w2 .=. w3)
    # CILINDRICAL_DIM = 'cilindrical_dim'  # dimensionalità forma cilindrica (w1 .=. w2 >> w3)
    # PLANAR_DIM = 'planar_dim'  # dimensionalità forma planare (w1 >> w2 .=. w3)
    # FA = 'fa'  # fractional anisotropy (0-> isotropic, 1-> max anisotropy
    # LOCAL_DISARRAY = 'local_disarray'   # local_disarray
    # LOCAL_DISARRAY_W = 'local_disarray_w'  # local_disarray using FA as weight for the versors


class Bcolors:
    V = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'



def plot_quiver_2d_for_save(x0_c, x1_c, x0_q, x1_q, img=None, shape=None, origin='upper', title='',
                            color_values='r', scale_units='xy', scale=1., width=2, pivot='middle', real=False,
                            color_map=None, cmap_image='gray', _show_plot=False):
    # xc, yc -> array of coordinates of TAIL of the Arrow
    # xq, yq -> array of components of quiver (Head of artow relative to tail)
    # real -> plot quiver with real xy dimension

    fig = plt.figure(tight_layout=True)


    # plotto tiff sotto i quiver
    if img is not None:
        plt.imshow(img, origin=origin, cmap=cmap_image, alpha=0.9)
        if shape is None:
            shape = img.shape
    else:
        color = 'k'  # background uniforme

    # rimuovo gli assi per ridurre la cornice bianca quando salvo l'immagine
    ax = plt.gca()
    ax.set_axis_off()
    fig.add_axes(ax)

    # plot all quivers
    if real:
        quiv = plt.quiver(x0_c, x1_c, x0_q, x1_q, headlength=0, headwidth=1, color=color)
    else:
        quiv = plt.quiver(x0_c, x1_c, x0_q, x1_q, color_values,
                          cmap=color_map,
                          units='xy', headwidth=1, headlength=0, width=width,
                          scale_units=scale_units, scale=scale, pivot=pivot)
    if _show_plot:
        plt.show()

    return fig, plt


######################################################################
###########################   MAIN   #################################
######################################################################

def main():

    # define parser
    parser = argparse.ArgumentParser(
        description='Save Strip frames with orientation quiver superimposed',
        epilog='Author: Francesco Giardini <giardini@lens.unifi.it>',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # command line REQUIRED argument
    parser.add_argument('-sf', '--source_folder', nargs='+', required=True, help='input images path')
    parser.add_argument('-r', '--orientation_numpy_filename', nargs='+', required=True, help='Orientation R numpy filename')

    # read parameters
    args = parser.parse_args()

    # extract path string
    source_path = manage_path_argument(args.source_folder)
    source_path = source_path.replace(' ', '\ ')  # correct whitespace with backslash
    base_path = os.path.dirname(os.path.dirname(source_path))
    stack_name = os.path.basename(source_path)

    # orientation file
    R_filename = args.orientation_numpy_filename[0]  # list of strings with len = 1

    # parameters file
    parameter_filepath = os.path.join(base_path, 'parameters.txt')
    ######################################################################

    print(Bcolors.OKBLUE + ' == Plotting and save orientation quiver over sample frames == ' + Bcolors.ENDC)
    print('* INPUT:')
    print('- Base path :\n     ', base_path)
    print('- Stack name:\n     ', stack_name)
    print('- R filename:\n     ', R_filename)
    print('- Parameters:\n     ', parameter_filepath)

    # extract parameters
    param_names = ['num_of_slices_P',
                   'sarc_length',
                   'res_xy', 'res_z',
                   'threshold_on_cell_ratio',
                   'threshold_on_peak_ratio',
                   'sigma']

    param_values = search_value_in_txt(parameter_filepath, param_names)

    # create dictionary of parameters
    parameters = {}
    for i, p_name in enumerate(param_names):
        parameters[p_name] = float(param_values[i])

    # Evaluate image analysis characteristics

    # analysis block dimension in z-axis
    num_of_slices_P = parameters['num_of_slices_P']

    # Parameters of Acquisition System:
    res_z = parameters['res_z']
    res_xy = parameters['res_xy']
    resolution_factor = res_z / res_xy
    print('\n * Acquisition info:')
    print('Pixel size (Real) in XY : {} um'.format(res_xy))
    print('Pixel size (Real) in Z : {} um'.format(res_z))

    # dimension of analysis block (parallelogram)
    block_side = int(num_of_slices_P * resolution_factor)
    shape_P = np.array((block_side, block_side, num_of_slices_P)).astype(np.int32)
    print('Dimension of Parallelepiped : {} pixel'.format(shape_P))
    print('Dimension of Parallelepiped : {} um'.format(np.array(shape_P) * np.array([res_xy, res_xy, res_z])))

    # load R
    print(R_filename)
    R_filepath = os.path.join(base_path, R_filename)
    R = np.load(R_filepath)
    shape_R = R.shape
    print('shape_R: ', shape_R)

    # scalar parameter to plot as color
    param_for_plot = None # choice from class Param

    # Normalize scalar parameter
    if param_for_plot is not None:
        R[param_for_plot] = normalize(R[param_for_plot].copy(), max_value=1.0, dtype=R[param_for_plot].dtype)

    # OPEN STACK --------------------------------------------------------------------------------
    t0 = time.time()

    # extract data (OLD METHOD)
    volume = load_tif_data(source_path)
    if len(volume.shape) == 2:
        volume = np.expand_dims(volume, axis=2)  # add the zeta axis

    t1 = time.time()

    # calculate dimension
    shape_V = np.array(volume.shape)
    pixels_for_slice = shape_V[0] * shape_V[1]
    total_voxel_V = pixels_for_slice * shape_V[2]
    print('\n* OPEN TIFF')
    print('Entire Volume dimension:')
    print('Volume shape     (r, c, z) : ({}, {}, {})'.format(shape_V[0], shape_V[1], shape_V[2]))
    print('Pixel for slice            : {}'.format(pixels_for_slice))
    print('Total voxel in Volume      : {}'.format(total_voxel_V))
    print('\n')
    print(' 3D stack readed in {0:.3f} seconds'.format(t1 - t0))

    print_info(volume, text='Volume')  # ----------------------------------------------------------

    # ================================================================================================================
    # =============================== PLOT PARAMETERS ================================================================
    # ================================================================================================================

    # ATTENZIONE - CICLO PER SALVARE LE SLICE CON I QUIVER SUPERIMPOSED - SE (save_all_fig = True) DURA MOLTO TEMPO!

    # savings - choice only ONE mode!
    _save_all_frames = False  # save every frame of tiff file with the corrispondent R depth vectors (VERY time expensive)
    _save_manual_fig = False  # save only manual selected depth in 'img_format' format selected (time expensive)
    _save_all_R_planes = True  # save one images for every R planes

    _show_plot = False  # display on QT windows the image created
    _plot_on_MIP = False
    _save_on_MIP = False

    # choice Z of R to plot
    if _save_manual_fig:
        #     manual_z_R_selection = range(1, 65, 3)
        # manual_z_R_selection = [0,1,2,3,4]
        manual_z_R_selection = None

    # choice what plot and what color_map
    color_to_use = COL_ZETA  # COL_XYANGLE, COL_PARAM, COL_ZETA
    color_map = cm.autumn
    _blur_par_to_plot = False  # gaussian blur on the color matrix

    # black or white background
    image_white = False  # True: LDG gray_R; False: LDS gray (normal)

    # equalize image - TODO
    # _equalize = True

    # image format to save (is 'save_fig' is selected):
    img_format = IMG_TIFF

    # Select index of eigenvector to plot (0: max, 2:min)
    ev_index = 2

    # quivers info to plot over images
    info_to_plot = 'none'  # option: 'ids', 'cell_ratio','cilindrical_dim','planar_dim',
    #                                'strenght', 'none', 'local_disarray', 'ew'

    # scale of quiver lenght
    scale = 0.4  # val più piccolo, quiver + lunghi


    # ================================================================================================================
    # =============================== END PARAMETERS =================================================================
    # ================================================================================================================

    # auto set parameters and advertising
    print('\n\n')
    if _save_all_frames:
        print(Bcolors.WARNING + '* ATTENTION : saving ALL frames -> WAIT NOTIFICATION of END OF PROCESS \n' + Bcolors.ENDC)
    else:
        print(
            Bcolors.OKBLUE + '* START creating plots...\n\n' + Bcolors.ENDC)

    if image_white:
        cmap_image = 'gray_r'
    else:
        cmap_image = 'gray'

    # ==== BLOCKS EXTRACTION FROM R AT EVERY Z PLANES =====

    # for each z in R, extract only cubes with freq info and their 'param_to_plot' values, and put in two list

    # extract bool maps of valid blocks
    orient_info_bool = R[Param.FREQ_INFO]  # TODO -> pensare a 'mappatura' di grane 'valide e non valide' tipo segmentazione

    # per ogni z di R, estraggo i cubi validi (utilizzando la mappa booleana orient_info_bool)
    # e li inserisco in una lista (Rf_z) alla z corrispondente
    Rf_z = list()
    param_to_plot_2d_list = list()

    # =============================== CREATING DATA FOR PLOT ()
    for z in range(shape_R[2]):

        # extract map of valid cells
        Rf_z.append(R[:, :, z][orient_info_bool[:, :, z]])  # R[allR, allrC, Z][bool_map_of_valid_blocks[allR, allC, Z]]

        # extract param_to_plt
        if _blur_par_to_plot:

            # extract values
            par_matrix = R[:, :, z][param_for_plot]
            # blurring
            par_blurred = ndimage.gaussian_filter(par_matrix.astype(np.float32), sigma=1.5).astype(np.float16)
            param_to_plot_2d_list.append(par_blurred[orient_info_bool[:, :, z]])

        else:
            param_to_plot_2d_list.append(R[:, :, z][orient_info_bool[:, :, z]][param_for_plot])



    # define real z to plot in volume system (effective slices = tiff frames)
    # z_R --> in 'R' riferiment system
    # z_vol --> in 'volume' riferiment system

    if _save_all_frames or _save_all_R_planes:
        z_R_to_plot = range(shape_R[2])  # all z in R matrix
    else:
        if manual_z_R_selection is not None:
            z_R_to_plot = manual_z_R_selection  # manual selection
        else:
            print(Bcolors.WARNING + 'manual_z_R_selection is None! - You have to select manually R plane to plot!' + Bcolors.ENDC)

    ###########################################################################################
    ###########################  START ITERATION TO DISPLAY FIGURES ###########################
    ###########################################################################################

    # elaborate every frames of R selected for the plot
    count = 0
    for z_R in z_R_to_plot:

        if not _plot_on_MIP:
            print(' - selected slice in R :      {} on {}'.format(z_R + 1, shape_R[2]))

        # SE QUESTO PIANO DI VETTORI NON E? VUOTO...
        if Rf_z[z_R].shape != (0,):
            print(' --- numbers of vector: ', Rf_z[z_R].shape)

            # extract position coordinate of every block
            init_coord = Rf_z[z_R][Param.INIT_COORD]  # rcz=yxz

            # extract quiver position
            centers_z = init_coord + (shape_P / 2)  # rcz=yxz
            yc_z = centers_z[:, :, 0]
            xc_z = centers_z[:, :, 1]
            zc_z = centers_z[:, :, 2]

            # extract quiver component
            quiver_z = Rf_z[z_R][Param.QUIVER_COMP]
            yq_z = quiver_z[:, :, 0]  # all y component
            xq_z = quiver_z[:, :, 1]  # all x component
            zq_z = quiver_z[:, :, 2]  # all z component

            # extract quiver ids
            ids = Rf_z[z_R][Param.ID_BLOCK]
            psd_ratio = Rf_z[z_R][Param.PSD_RATIO]

            # extract par values to plot
            param_to_plot_z = param_to_plot_2d_list[z_R]

            # if not _plot_on_MIP:
            #     print('z_R: {}  ->  Rf_z[z_R][param_for_plot].shape: {}'.format(z_R, Rf_z[z_R][param_for_plot].shape))

            # create color map
            if color_to_use is COL_PARAM:
                color_values = param_to_plot_z
            elif color_to_use is COL_XYANGLE:
                color_values = (np.arctan2(yq_z, xq_z) % np.pi)[:]  # [:] because has shape (n_blocks, 1)
                color_values = 2 * np.abs((color_values / color_values.max()) - 0.5)  # simmetric map of angles
            elif color_to_use is COL_ZETA:
                color_values = normalize(np.abs(zq_z), max_value=1.0, dtype=np.float64)  # norm between [0,1]
                # zeta no simmetrica?
                # color_values = 2 * np.abs((color_values / color_values.max()) - 0.5)  # simmetric

            # maps color_values into scalar of 'color_map' matplotlib color map
            colors_2d = float_to_color(values=color_values, color_map=color_map)

            # create range of real frames of volume to plot
            if _save_all_frames:
                slices_to_plot = range(z_R * shape_P[2], (z_R + 1) * shape_P[2])  # plotta tutte le 8 slide per ogni cubetto
            elif _save_all_R_planes or _save_manual_fig:
                slices_to_plot = [((z_R + 1 / 2) * shape_P[2]).astype(int)]  # plotta solo la slide centrale

            if _plot_on_MIP:

                print('     plot on MIP')
                MIP = normalize(np.max(volume, axis=2), dtype=np.uint8, max_value=100)
                # shape_fig = MIP.shape  # image shape

                width = 5
                # prepare plot for save and/or plot image
                fig, plt = plot_quiver_2d_for_save(xc_z, yc_z, -xq_z, yq_z, color_values=color_values, img=MIP,
                                                   title='MIP - ev: {0}th'.format(ev_index),
                                                   scale_units='xy', scale=scale, pivot='middle',
                                                   real=False, width=width,
                                                   color_map=color_map, cmap_image=cmap_image,
                                                   _show_plot=_show_plot)
                plt.show()

                if _save_on_MIP:
                    # TIFF
                    # (1) save the image in memory in PNG format
                    png1 = BytesIO()
                    fig.savefig(png1, format='png')
                    # (2) load this image into PIL
                    png2 = Image.open(png1)
                    # (3) save as TIFF
                    png2.save(os.path.join(base_path, 'quiver_on_MIP.tiff'))
                    png1.close()
                    print("saved fig MIP in ", base_path)

            else:

                for z_vol in slices_to_plot:

                    # check depth of selected z frame
                    if z_vol >= shape_V[2]:
                        # take the last one
                        z_vol = shape_V[2] - 1

                    # extract frame
                    img_z = normalize(volume[:, :, z_vol], dtype=np.uint8, max_value=100)
                    print('     selected slice in Volume : {} on {} \n'.format(z_vol, shape_V[2]))

                    # if _equalize :
                    # TODO

                    # shape_fig = volume[:, :, z_vol].shape  # image shape

                    # ATTENZIONE   HO AGGIUNTO IL  MENO  ALLA  X   <-----------------------! ! ! !

                    # prepare plot for save and/or plot image
                    fig, plt = plot_quiver_2d_for_save(xc_z, yc_z, -xq_z, yq_z, color_values=color_values, img=img_z,
                                                       title='R:{0} - Z:{1} - ev: {2}th'.format(z_R, z_vol, ev_index),
                                                       scale_units='xy', scale=scale, pivot='middle',
                                                       real=False, width=3,
                                                       color_map=color_map, cmap_image=cmap_image)

                    # [if selected] plot block info over image for every vector
                    if info_to_plot is not 'none':
                        if info_to_plot is 'ids':
                            to_write = ids
                        if info_to_plot is 'ew':
                            to_write = Rf_z[z_R][info_to_plot][:, 0, ev_index]  # [all_blocks, {è in riga}, indice_ew]
                            print('-------------------------------')
                            print(to_write.shape)
                            print(to_write)
                            print('-------------------------------')
                        if info_to_plot in ['cell_ratio', 'cilindrical_dim', 'planar_dim', 'strenght']:
                            to_write = Rf_z[z_R][info_to_plot]
                            print('-------------------------------')
                            print(to_write.shape)
                            print(to_write)
                            print('-------------------------------')

                        for val, pos in zip(to_write, init_coord):
                            r = pos[0][0]  # shape is (1, rcz)
                            c = pos[0][1]  # shape is (1, rcz)
                            string = str(val) if info_to_plot is 'ids' else '{0:.1f}'.format(val)
                            plt.text(c, r, string, color='r', fontsize=10)
                            plt.title(R_filename + ' ' + info_to_plot + ' ' + R_filename)
                        plt.show()

                    # saving images?
                    if _save_all_frames or _save_manual_fig or _save_all_R_planes:
                        quiver_path = os.path.join(base_path, 'quiver_angle_{}_{}/'.
                                                   format(img_format,
                                                          R_filename.split('.')[0]))  # create path where save images
                        # check if it exist
                        if not os.path.isdir(quiver_path):
                            os.mkdir(quiver_path)

                        # create img name
                        img_name = str(z_vol) + '_ew{}'.format(ev_index)

                        if img_format == IMG_SVG:
                            # formato SVG -> puoi decidere dopo la risoluzione aprendolo con fiji
                            fig.savefig(str(quiver_path + img_name + '.svg'), format='svg',
                                        dpi=1200, bbox_inches='tight', pad_inches=0)

                        elif img_format == IMG_EPS:
                            # formato EPS buono per latex (latex lo converte automat. in pdf)
                            fig.savefig(str(quiver_path + img_name + '_black.eps'), format='eps', dpi=400,
                                        bbox_inches='tight', pad_inches=0)

                        elif img_format == IMG_TIFF:
                            # (1) save the image in memory in PNG format
                            png1 = BytesIO()
                            fig.savefig(png1, format='png')
                            # (2) load this image into PIL
                            png2 = Image.open(png1)
                            # (3) save as TIFF
                            png2.save((str(quiver_path + img_name + '.tiff')))
                            png1.close()

                        plt.close(fig)
                        count += 1
        # piano di R vuoto
        else:
            print(' --- numbers of vector: ', Rf_z[z_R].shape, ' frame not displayed\n')
            count += 1

    print('\n ** Process finished - OK')


if __name__ == '__main__':
    main()
