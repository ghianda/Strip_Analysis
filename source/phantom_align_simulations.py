# S.R. yxz = (row, col, zeta)
#
# POLAR COORDINATES in ISO convention (r, θ, φ)
# r >= 0              radial distance (also 'rho')
# -180 < φ <= +180    azimuth angle ('phi')  -> on plane xy
# 0 <= θ <= +180      polar angle ('theta')  -> from z axis (below) up to vector

# NB - 'ANGLES' = (theta, phi)

# x le STRIP:
# le orientazioni sono swappate nella semisfera y>0
# non becco inclinazioni maggiorni di 30 gradi
# ALLORA
# -90 < φ <= +90  (NB - -90 == +90)
# -30 <= θ <= +30

import os
import time
import shutil
import argparse
import inspect

import numpy as np
import matplotlib.pyplot as plt
import tifffile
import seaborn as sns
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)

# rotation tools
from scipy.ndimage.interpolation import rotate as scipy_rotate
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import zoom
from scipy.ndimage.filters import gaussian_filter1d
from scipy.ndimage.filters import gaussian_filter, median_filter

from zetastitcher import InputFile

from custom_tool_kit import seconds_to_hour_min_sec, print_time_execution, all_words_in_txt
from custom_image_tool import create_img_name_from_index
from make_data import manage_path_argument

# strip analysis
from GAMMA_orientation_analysis_no_outlier import block_analysis, extract_parameters, create_3D_filter


def blur_on_z_axis(vol, sigma_px=1):
    # Opera uno smoothing 1d lungo l'asse z:
    # itera in y selezionando i piani XZ, e applica su ogni piano XZ
    # un gaussian blur 1d lungo l'asse 1 (Z)
    # con un kernel gaussiano di sigma = sigma_px (in pixel)

    smoothed = np.zeros_like(vol)

    for y in range(vol.shape[0]):
        smoothed[y, :, :] = gaussian_filter1d(input=vol[y, :, :], sigma=sigma_px, axis=1, mode='reflect')

    return smoothed


def sigma_from_FWHM(FWHM_um=1, px_size=1):
    # Legend - in the "Blurring" methods:
    # SD: Standard Deviation = sigma; Var: Variance = sigma**2
    #
    # a gaussian kernel with sigma = s depends by resolution (FWHM in um)
    #
    # This function calculates sigma (in micron) by FWHM
    # and return sigma in pixel using the given pixel sixe.

    # estimate "variance" ( = sigma**2) of gaussian kernel by FWHM
    sigma2 = FWHM_um ** 2 / (8 * np.log(2))  # micron

    # SD (= sigma) of Gaussian Kernel
    sigma = np.sqrt(sigma2)  # micron

    # return SD in pixel
    return sigma / px_size



def save_quiver_plot(v, theta, phi, title=None, _display=True, _save=False, path=None, fig_filename='_empty_.fake'):
    # prepare plot 3d
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # position of vector YXZ
    pos = np.array([0, 0, 0])

    # dim of grid
    ax.set_xlim3d(0, 1)
    ax.set_ylim3d(0, 1)
    ax.set_zlim3d(-1, 1)

    # label ticks
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # default initial orientation of plot
    # azim : float, optional - Azimuthal viewing angle, defaults to -60.
    # elev : float, optional - Elevation viewing angle, defaults to 30.
    ax.elev = 210
    ax.azim = -110

    # create polar coordinates trings
    theta_str = 'θ={0:0.1f}'.format(theta)
    phi_str = 'φ={0:0.1f}'.format(phi)
    polar_string = theta_str + ', ' + phi_str

    # write polar coordinates on plot
    ax.text(0.6, 0.6, 0.6, polar_string, None)

    # plot quiver
    ax.quiver(pos[1], pos[0], pos[2], v[1], v[0], v[2], length=1, normalize=True)

    # fig title
    if title is None:
        plt.title('v(yxz) = [{0:0.2f}, {1:0.2f}, {2:0.2f}] (yxz)'.format(v[0], v[1], v[2]))
    else:
        plt.title(title)

    if _display:
        # display plot
        plt.show()

    # save figure
    if _save and path is not None:
        if fig_filename is None:
            fig_filename = str(theta_str + '_' + phi_str) + '.pdf'
        fig.tight_layout()
        fig.savefig(os.path.join(path, fig_filename), bbox_inches='tight')

    if not _display:
        plt.close()


def polar_coordinates(v):
    # evaluate polar coordinates
    xy = v[1] ** 2 + v[0] ** 2
    rho = np.sqrt(xy + v[2] ** 2)  # radius
    theta = (180 / np.pi) * np.arctan2(np.sqrt(xy), v[2])  # for elevation angle defined from Z-axis down ()
    phi = (180 / np.pi) * np.arctan2(v[1], v[0])  # phi
    return (rho, theta, phi)


# define rotation of angle_zr around z axis:
def create_rotation_matrix(angle_in_deg, axis):
    # conversion in radians
    rad = angle_in_deg * np.pi / 180

    if axis in [1, 2]:

        # around 1 axis (X)
        if axis == 1:
            rot = R.from_matrix([[np.cos(rad), 0, np.sin(rad)],
                                 [0, 1, 0],
                                 [-np.sin(rad), 0, np.cos(rad)]])

        # around 2 axis (Z)
        if axis == 2:
            rot = R.from_matrix([[np.cos(rad), -np.sin(rad), 0],
                                 [np.sin(rad), np.cos(rad), 0],
                                 [0, 0, 1]])

        return rot
    else:
        return None


def cartesian_coordinates(rho, theta, phi):
    ''' NB - (phi, theta) in degree (0, 360)'''

    # conversion in radiant
    theta_rad = theta * np.pi / 180
    phi_rad = phi * np.pi / 180

    x = rho * np.sin(theta_rad) * np.sin(phi_rad)
    y = rho * np.sin(theta_rad) * np.cos(phi_rad)
    z = rho * np.cos(theta_rad)

    return (y, x, z)


def rotate_volume(vol, angle_in_deg, axis, mode='wrap'):
    # select plane of rotation:
    # around 1 axis (X)
    if axis == 1: axes = (2, 0); angle_in_deg = -angle_in_deg  # (around X axis) - anti-coherent with theta convention
    if axis == 2: axes = (0, 1)  # (around Z axis) - coherent with phi convention

    # create rotated volume
    rotated = scipy_rotate(input=vol, angle=angle_in_deg, axes=axes, reshape=False, output=None, mode=mode)

    # mode : str, optional
    # Points outside the boundaries of the input are filled according to the given mode:
    # {‘reflect’, ‘constant’, ‘nearest’, ‘mirror’, ‘wrap’}, optional

    return rotated


class ERR_FLAG:
    APPLIED = 'applied'
    V0 = 'v0'
    VR = 'vr'
    VEST = 'v_est'
    POLAR_V0 = 'polar_v0'
    POLAR_VR = 'polar_vr'
    POLAR_VEST = 'polar_v_est'
    ERROR = 'error'


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


def extract_orientation_vector(vol, param, mask=None, method='fft'):
    # PER ADESSO SOLO PER LE STRIP
    if method == 'fft':

        # spatial analysis resolution
        num_of_slices_P = param['num_of_slices_P']
        resolution_factor = param['res_z'] / param['res_xy']
        block_side = int(num_of_slices_P * resolution_factor)
        shape_P = np.array((block_side, block_side, num_of_slices_P)).astype(np.int32)

        # create frequency filter
        if mask is None:
            mask = create_3D_filter(block_side, param['res_xy'], param['sarc_length'])

        # fft orientation analysis
        there_is_cell, there_is_freq, results = block_analysis(vol,
                                                               shape_P,
                                                               param,
                                                               block_side,
                                                               mask,
                                                               verbose=True,
                                                               lines=[])

        # extract orientation from results
        v = np.array(results['quiver_comp'])

        # normalize vector
        v = v / np.linalg.norm(v)

        # turn with y > 0
        if v[0] < 0:
            v = -v

        return v
    else:
        return None


def apply_3d_rotations(vol, theta=0, phi=0, res_xy=1, res_z=1, mode='wrap'):
    # apply a rotation to an isotropic version of "vol" and
    # return the rotated volume with the original pixel size

    # make vol isotropic
    res_factor = res_z / res_xy
    vol_isotropic = zoom(vol, (1, 1, res_factor))

    # apply rotations
    vol_iso_rot_temp = rotate_volume(vol=vol_isotropic, angle_in_deg=theta, axis=1, mode=mode)  # theta rotation
    vol_iso_rotated = rotate_volume(vol=vol_iso_rot_temp, angle_in_deg=phi, axis=2, mode=mode)  # phi rotation

    # rescale the rotated volume to the original pixel size
    vol_rotated = zoom(vol_iso_rotated, (1, 1, 1 / res_factor))
    return vol_rotated



def plot_error_heatmap(error_mtrx, k=1, _display=True, _save_plot=False, _plotpath=None, ann_size=10, fmt='.2g'):
    # extract virtual rotation applied
    theta_list = list(error_mtrx[ERR_FLAG.APPLIED][:, :, 0][:, 0])
    phi_list = list(error_mtrx[ERR_FLAG.APPLIED][:, :, 1][0, :])

    ax = sns.heatmap(k * error_mtrx[ERR_FLAG.ERROR], linewidth=0.5, annot=True, annot_kws={"size": ann_size},
                     xticklabels=phi_list, yticklabels=theta_list, fmt=fmt)  # error module

    ax.set_xlabel('Phi (degree)')
    ax.set_ylabel('Theta (degree)')
    plt.title('Error % between V_estimated and V_rotated (|Ve - Vr|)')

    if _display:
        plt.show()

    if _save_plot and _plotpath is not None:
        # save figure
        figname = os.path.join(_plotpath, 'Error_ve_vr_heatmap' + '.pdf')
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(figname, bbox_inches='tight')

    return None


def rotate_sample(vol, theta_deg=0, phi_deg=0, res_xy=1, res_z=1, mode='wrap', FWHM_um=3.1, _smooth=True):
    # create new rotated volume
    block_rotated = apply_3d_rotations(vol=vol, theta=theta_deg, phi=phi_deg,
                                       res_xy=res_xy, res_z=res_z,
                                       mode=mode)

    if _smooth:
        # smooth on z axis to simulate FWHMz of acquisition system
        sigma_z_px = sigma_from_FWHM(FWHM_um=FWHM_um, px_size=res_z)
        block_ready = blur_on_z_axis(vol=block_rotated, sigma_px=sigma_z_px)
    else:
        block_ready = block_rotated.copy()

    return block_ready



def linear_plot(x, y1, y2=None, label1='label1', label2='', kx=[], ky=[], title='', xlabel='', ylabel='',
                legend_loc='upper left', _save=False, filepath=None):
    # create plot
    fig, ax = plt.subplots()

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # plot y1
    ax.plot(x, y1, label=label1)

    # plot y2 (if exist)
    if y2 is not None:
        ax.plot(x, y2, label=label2)

    # draw vertical lines (if passed)
    for k in kx:
        ax.axvline(x=k, color='k', linestyle='--')

    # draw horizontal lines (if passed) from 0 to the last vertical line (if passed, else: max)
    for k in ky:
        if ky == []:
            ax.axhline(y=k, color='k', linestyle='--')
        else:
            ax.axhline(y=k, xmax=kx[-1], color='k', linestyle='--')

    # Make a plot with major ticks that are multiples of 20 and minor ticks that
    # are multiples of 5.  Label major ticks with '%d' formatting but don't label
    # minor ticks.
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

    # For the minor ticks, use no labels; default NullFormatter.
    ax.yaxis.set_minor_locator(MultipleLocator(1))

    # plot legend and show fig
    ax.legend(loc=legend_loc)
    plt.show()

    # save
    if _save and filepath is not None:
        fig.tight_layout()
        fig.savefig(filepath, bbox_inches='tight')


def extract_subvol(vol, shape_subvol=None):
    # extract sub-volume with shape = shape_subvol from the volume 'vol'.
    # the sub-volume is defined (for each axis) as:
    # [0, ..., a ----- b, ..., end] in the 'big' reference system
    # for each axis i = 0,1,2, the "borders" are:
    # |0,ai| = |bi,end| = int_sup((side_i_big - side_i_soul) / 2)

    if shape_subvol is None:
        return vol
    else:
        shape_vol_px = vol.shape

    # extract border size for each axis:
    ai = np.zeros(3).astype(np.int)
    bi = np.zeros(3).astype(np.uint64)
    for i in [0, 1, 2]:
        ai[i] = int(np.ceil((shape_vol_px[i] - shape_subvol[i]) / 2))
        bi[i] = int(ai[i] + shape_subvol[i])

    # extract sub-volume |0... [ai, bi] ...end| from big volume
    return np.copy(vol[ai[0]:bi[0], ai[1]:bi[1], ai[2]:bi[2]])


def fill_vol_with_rep(shape_bigvol, source_vol, _blur=False, border=3, filter_size=3):
    # repeat source_vol in the 3d space to fill a bigger volume
    # if _blur is True, smooth the intersection with a 3d gaussian kernel

    if np.any(shape_bigvol > source_vol.shape):

        # evaluate number of ripetition of each axis (int_sup)
        rip = np.zeros(3)
        for ax in [0, 1, 2]:
            rip[ax] = np.ceil(shape_bigvol[ax] / source_vol.shape[ax])

        # create volume with repeated version of source_vol
        repeated = np.tile(source_vol, (int(rip[0]), int(rip[1]), int(rip[2])))

        if _blur:
            # blur in the intersection
            repeated = smooth_intersection(repeated, source_vol.shape,
                                           border=border, filter_size=filter_size)

        # crop the desidered shape
        cropped = np.copy(repeated[0: shape_bigvol[0],
                          0: shape_bigvol[1],
                          0: shape_bigvol[2]])

        return cropped
    else:
        # source_vol is bigger than bigvol
        return source_vol


def smooth_intersection(vol, subvol_shape, border=1, filter_size=2):
    # for each axis (axis=0), apply a 2d-median filter along the borders (axis=1)
    # of the sub-volume in all the planes (axis=2).
    # Change the value of 'b' pixels around the intersection (-b, +b)
    # and use a median filter with size = filter_size.
    #
    # To iterate in the three axis, swap the axis in this way:
    # iter=1: YXZ (border along Y, smooth in the YX plane, iter planes in Z)
    # iter=2: XYZ (border along X, smooth in the YX plane, iter planes in Z)
    # iter=1: ZXY (border along X, smooth in the XZ plane, iter planes in Y)
    #
    # at the end, restore the YXZ convention

    # evaluate number of intersection of each axis (int_sup)
    inters = np.zeros(3)
    for ax in [0, 1, 2]:
        inters[ax] = np.ceil(vol.shape[ax] / subvol_shape[ax]) - 1

    # create a copy of source volume
    v = np.copy(vol)

    # for each axis
    for ax in [0, 1, 2]:

        # move the current axis in the first position
        v = np.moveaxis(v, ax, 0)

        # for each intersection [1 -> # inters]
        for i in range(1, int(inters[ax]) + 1):

            # define intersection position along current axis
            int_pos = i * subvol_shape[ax]

            # smooth the selected intersection along current axis on each planes
            for plane in range(0, v.shape[2]):
                v[int_pos - border: int_pos + border, :, plane] = median_filter(
                    np.copy(v[int_pos - border: int_pos + border, :, plane]), size=filter_size)

                # restore original axis order
    v = np.swapaxes(v, 2, 0)  # ZXY -> YXZ
    return v


def create_phantom_align(esp_zero, param, angles=(0, 0), shape_soul_px=np.array([182, 45, 10]),
                         shape_phantom_px=None, mask=None, _blur_inters=True, border=3, filter_size=3):
    # take time
    # start = time.time()

    # define phantom dimension
    # if mask if passed, phantom shape = mask shape, else shape_phantom_px is passed
    # if bot mask and shape_phantom_px are None, return the esp_zero volume
    if mask is not None:
        shape_phantom_px = mask.shape
    else:
        if shape_phantom_px is None:
            return esp_zero  # return the whole volume rotated

    # create rotated version of expanded volume
    esp_rotated = rotate_sample(vol=esp_zero,
                                theta_deg=angles[0], phi_deg=angles[1],
                                res_xy=param['res_xy'], res_z=param['res_z'])

    # extract soul from rotated volume
    soul = extract_subvol(vol=esp_rotated, shape_subvol=shape_soul_px)

    # fill the phantom sample with many copy of "soul" vol
    phantom = fill_vol_with_rep(shape_bigvol=shape_phantom_px, source_vol=soul,
                                _blur=_blur_inters, border=border, filter_size=filter_size)

    # take time execution
    # end = time.time()
    # print_time_execution(end=end, start=start)

    # if mask is passed, segment the phantom
    if mask is not None:
        return phantom * mask.astype(np.bool)
    else:
        return phantom


# modality of creation of phantom
class MOD_PHANTOM:
    ALL = 1
    DIS = 2


########################################    MAIN    ##########################################
def main():

    parser = argparse.ArgumentParser(
        description='Create and analyze a pool of phantom samples',
        epilog='Author: Francesco Giardini <giardini@lens.unifi.it>',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # command line arguments
    parser.add_argument('-st', '--source_tiff', nargs='+', required=True, help='filepath of espansed source tiff')
    parser.add_argument('-m', '--mask', nargs='+', required=False, default=None,
                        help='filepath of tiff mask for segmentation (optional)')
    parser.add_argument('-k', action='store_true', default=False, dest='keep',
                        help='keep phantom tiff files (default -> delete files)')
    parser.add_argument('-b', action='store_true', default=False, dest='blur',
                        help='smooth intersection in the phantom with a median filter')
    parser.add_argument('-t', action='store_true', default=False, dest='test',
                        help='use \'test\' configuration (small phantom, few sigmas)')

    # parse arguments
    args = parser.parse_args()

    # extract path strings
    filepath = manage_path_argument(args.source_tiff)
    maskpath = manage_path_argument(args.mask) if args.mask is not None else None
    _keep        = args.keep
    _blur_inters = args.blur
    _test        = args.test

    # correct whitespace with backslash
    # filepath = filepath.replace(' ', '\ ')
    # maskpath = maskpath.replace(' ', '\ ')

    # extract basepath
    basepath   = os.path.dirname(filepath)

    #===========================================================
    # ==================    SETTINGS     =======================
    # ==========================================================

    # parameters filename
    par_filename = 'parameters.txt'

    # used if mask is not passed
    shape_phantom_manual_um = np.array([500, 250, 250]) if not _test else np.array([300, 150, 150])

    # soul shape
    shape_soul_um = np.array([80, 20, 20])  # shape of 3d soul

    output_folder = 'phantoms_blur_intersections' if _blur_inters else 'phantoms_no_blur_inters'
    orientation_script_name = 'extract_orientations.py'
    results_filename = 'GAMMA_orientation_results.txt'

    # Parameters of blurring filter in the intersection (if '-b' is passed)
    # ADVICE: virtual_sample => False, real_sample => True
    b = 3  # pixel selecetd on both side [-b, +b] around intersection
    size = 3  # sigma of median filter

    _plot = False

    # define (theta, phi) to be tested
    if _test:
        angles_range = [[-10, -5, 0, 5, 10], [-10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10]]  # ONLY FOR TEST
    else:
        angles_range = [range(-30, 35, 5), range(-30, 35, 5)]  # (theta, phi) # ARTICOLO
        # angles_range = [[0], [0]]  # (theta, phi) # test mio
    #===========================================================
    #===========================================================

    print('\n*** INPUT : ')

    print('- Tiff file for elementary volume: \n', filepath)
    if maskpath is not None: print('- Tiff file of real sample segmentation: \n', maskpath)

    # parameteres filepath
    par_filepath = os.path.join(basepath, par_filename)
    print('- Parameters file: \n', par_filepath)

    # create dinamically path of extract_orientations.py script
    script_basepath = os.path.dirname(os.path.abspath(inspect.getabsfile(inspect.currentframe())))
    orientation_script_fullpath = os.path.join(script_basepath, orientation_script_name)
    print('- Orientation script path: \n', orientation_script_fullpath)

    # --- LOADING -------------------------------------------

    # load tif of espanse volume and move axis to YXZ ref
    esp_zero = np.moveaxis(InputFile(filepath).whole(), 0, -1)
    print('\n- Source Tiff loaded.')
    if _plot: plt.figure(), plt.imshow(esp_zero[..., 0]), plt.title('esp_zero[..., 0]')

    if maskpath is not None:
        # load tif of segmentation and move axis to YXZ ref
        mask = np.moveaxis(InputFile(maskpath).whole(), 0, -1)
        print('\n- Mask Tiff loaded.')
    else:
        mask = None
        print('\n- No mask selected for segmentation')

    # reads parameters (strip)
    param = extract_parameters(par_filepath)
    pixel_size_yxz = np.array([param['res_xy'], param['res_xy'], param['res_z']])
    # --------------------------------------------------------

    # shapes
    shape_soul_px = (shape_soul_um / pixel_size_yxz).astype(np.uint)
    if mask is not None:
        shape_phantom_px = mask.shape
        shape_phantom_um = shape_phantom_px * pixel_size_yxz
    else:
        shape_phantom_um = shape_phantom_manual_um
        shape_phantom_px = (shape_phantom_um / pixel_size_yxz).astype(np.uint)

    # def number of total phantoms and results matrices
    n_phantom     = len(angles_range[0]) * len(angles_range[1])
    align_mtrx    = np.zeros((len(angles_range[0]), len(angles_range[1])))
    dis_area_mtrx = np.zeros((len(angles_range[0]), len(angles_range[1])))
    dis_std_mtrx  = np.zeros((len(angles_range[0]), len(angles_range[1])))

    print('-----------------------  -  ATTENTION  - ------------------------\n')
    print('--------   You will create {0:4.0f} panthom samples, ok?   ----------   '.format(n_phantom))
    if ~_keep:
        print('------------------ All tiff files will be deleted   -------------')
    print('-----------------------------------------------------------------)')
    print('Shape of phantom: ', shape_phantom_um, 'um ---> ', shape_phantom_px, 'px')
    print('Shape of soul   : ', shape_soul_um, 'um ---> ', shape_soul_px, 'px')

    # save elaboration time for each iteration (for estimate mean time of elaboration)
    phantom_elab_time_list = list()
    start_tot = time.time()
    count = 1

    print('\n Start creation and analysis of the series of phantom with (theta, phi) = \n')
    count = 1
    for (r, theta) in enumerate(angles_range[0]):
        for (c, phi) in enumerate(angles_range[1]):

            elab_start = time.time()

            # phantom creation
            print('  - n\' {} of {} - Creating and analyzing with (theta, phi) = ({},{})...'.format(count, n_phantom, theta, phi))
            phantom = create_phantom_align(esp_zero=esp_zero,
                                           param=param,
                                           angles=(theta, phi),
                                           shape_soul_px=shape_soul_px,
                                           shape_phantom_px=shape_phantom_px,
                                           mask=mask,
                                           _blur_inters=_blur_inters,
                                           border=b,
                                           filter_size=size)

            # define current paths
            sub_folder = 'analyze_phantom_{0:0.0f}_{1:0.0f}_'.format(theta, phi)
            seq_folder = 'segmented/stack'
            current_path = os.path.join(basepath, output_folder, sub_folder)

            # create current phantom folder
            if not os.path.isdir(os.path.join(current_path, seq_folder)):
                os.makedirs(os.path.join(current_path, seq_folder))

            # save as image sequence of tiff in the folder
            for z in range(phantom.shape[2]):
                img_z_name = create_img_name_from_index(z, pre='phantom_')
                tifffile.imsave(
                    os.path.join(current_path, seq_folder, img_z_name), phantom[..., z])

            # copy parameters.txt file
            shutil.copyfile(par_filepath, os.path.join(current_path, par_filename))

            # analyze the sample
            os.system('python3 {} -sf {} > {}'.format(
                orientation_script_fullpath,
                os.path.join(current_path, seq_folder),
                os.path.join(current_path, 'out_execution.txt')))

            # read current results from .txt
            results_txtpath = os.path.join(current_path, results_filename)
            words = all_words_in_txt(results_txtpath)

            # write the results in the matrices
            align_mtrx[r, c]    = words[words.index('Alignment') + 2]
            dis_area_mtrx[r, c] = words[words.index('(area_ratio)') + 2]
            dis_std_mtrx[r, c]  = words[words.index('dev.std)') + 2]

            if not _keep:
                # delete tiff files of current sample
                shutil.rmtree(os.path.join(current_path, seq_folder))

            phantom_elab_time_list.append(time.time() - elab_start)
            elapsed_time = (n_phantom - count) * np.mean(phantom_elab_time_list)
            (h, m, s) = seconds_to_hour_min_sec(elapsed_time)

            count = count + 1
            print(' ---   ET: {0:2d}h {1:2d}m {2:2d}s'.format(int(h), int(m), int(s)))

    # savings
    np.save(os.path.join(basepath, 'theta_tested.npy'), np.array(angles_range[0]))
    np.save(os.path.join(basepath, 'phi_tested.npy'), np.array(angles_range[1]))
    np.save(os.path.join(basepath, 'align_mtrx.npy'), align_mtrx)
    np.save(os.path.join(basepath, 'dis_area_mtrx.npy'), dis_area_mtrx)
    np.save(os.path.join(basepath, 'dis_std_mtrx.npy'), dis_std_mtrx)

    end_tot = time.time()
    print_time_execution(start=start_tot, end=end_tot, proc_name='\nCreated {} phantom - '.format(n_phantom))

    return None


if __name__ == '__main__':
    main()
