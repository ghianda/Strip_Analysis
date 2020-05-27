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

from custom_tool_kit import seconds_to_hour_min_sec, print_time_execution, all_words_in_txt, create_coord_by_iter, \
    create_slice_coordinate
from custom_image_tool import create_img_name_from_index, print_info
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
    ''' NB - (theta, phi) are in degree (0, 360)'''

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


def fill_with_random_rotations(phantom, esp_zero, shape_soul_px, ripetition_step, pixel_size, theta_dist, phi_dist,
                               blur=False, border=3, filter_size=3):
    # phantom is the big voulme, esp_zero is the source vol, shape_soul_px is the shape of the souls.
    # The function fill the phantom with N copy of souls extracted by the esp_zero, rotated randomly.
    # Theta_dist anf phi_dist are the random angles to be used to rotate esp_zero for each soul.
    # if _blur is True, smooth the intersection with a 3d gaussian kernel
    # ripetition_step are the number of iteration of each axis to fill phantom with souls

    if np.any(phantom.shape > shape_soul_px):

        # start iteration to compose the phantom
        # print('  - z = ', end='')
        for zr in range(ripetition_step[2]):
            # print('{0:0.0f} um ... '.format(zr * pixel_size[2]), end='')
            for yr in range(ripetition_step[0]):
                for xr in range(ripetition_step[1]):

                    # extract random angles for rotate local sample
                    theta = np.random.choice(theta_dist, 1)[0]
                    phi = np.random.choice(phi_dist, 1)[0]

                    # create rotated version of expanded volume
                    esp_rotated = rotate_sample(vol=esp_zero,
                                                theta_deg=theta, phi_deg=phi,
                                                res_xy=pixel_size[0], res_z=pixel_size[2])

                    # extract soul from rotated volume
                    soul = extract_subvol(vol=esp_rotated, shape_subvol=shape_soul_px)

                    # find current portion of phantom to fill with the soul
                    start_coord = create_coord_by_iter(yr, xr, zr, shape_soul_px)
                    start_coord = tuple(np.array(start_coord).astype(np.uint64))  # cast to int the current coordinates
                    slice_coord = create_slice_coordinate(start_coord, shape_soul_px)

                    # insert current soul in the right position in the phantom
                    try:
                        phantom[slice_coord] = np.copy(soul)
                    except:
                        print('[!] ---> ERROR <----')
                        print_info(soul, text='SOUL INFO:')
                        print_info(phantom, text='PHANTOM INFO:')
                        print('type(yr)       :', type(yr))
                        print('type(xr)       :', type(xr))
                        print('type(zr)       :', type(zr))
                        print('slice_coord    :', slice_coord)
                        print('start_coord    :', start_coord)
                        print('ripetition_step:', ripetition_step)
                        print('ripetition_step.dtype:', ripetition_step.dtype)
                        print('shape_soul_px  :', shape_soul_px)
                        print('shape_soul_px.dtype  :', shape_soul_px.dtype)
                        return None

        # [!] - now phantom is filled
        return phantom
    else:
        return None


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
    inters = np.zeros(3, np.uint64)
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
            # cast to int because np.int * np.uint64 --> np.float64
            int_pos = np.uint64(i * subvol_shape[ax])

            # smooth the selected intersection along current axis on each planes
            for plane in range(0, v.shape[2]):
                v[int(int_pos - border): int(int_pos + border), :, plane] = median_filter(
                    np.copy(v[int(int_pos - border): int(int_pos + border), :, plane]), size=filter_size)
                # try:
                #     v[int_pos - border: int_pos + border, :, plane] = median_filter(
                #         np.copy(v[int_pos - border: int_pos + border, :, plane]), size=filter_size)
                # except:
                #     print('[!] ---> ERROR <----')
                #     print_info(v, text='VOLUME INFO:')
                #     print('type(ax)       :', type(ax))
                #     print('type(i)        :', type(i))
                #     print('type(int_pos)  :', type(int_pos))
                #     print('shape_soul_px        :', subvol_shape)
                #     print('shape_soul_px.dtype  :', subvol_shape.dtype)
                #     return None

                # restore original axis order
    v = np.swapaxes(v, 2, 0)  # ZXY -> YXZ
    return v


def create_phantom_disarray(esp_zero, param, sigma_theta, sigma_phi, mu_angles=(0, 0),
                            shape_soul_px=np.array([182, 45, 10]),
                            shape_phantom_px=None, mask=None, _blur_inters=True, filter_border=3, filter_size=3):
    # create a phantom and fill it with N copy of souls.
    # Each soul is obtained by a rotation of esp_zero.
    # Ttation angles(theta, phi) are extracted by two random gaussian distributions with sigma passed as arguments.
    # If mask is passed, shape of phantom is equel of shape of mask.

    # define phantom dimension
    # if mask if passed, phantom shape = mask shape, else shape_phantom_px is passed
    # if bot mask and shape_phantom_px are None, return the esp_zero volume
    if mask is not None:
        shape_phantom_px = mask.shape
    else:
        if shape_phantom_px is None:
            return esp_zero  # return the whole volume rotated

    # evaluate number of ripetition of each axis (int_sup)
    rip = np.zeros(3).astype(np.uint64)
    for ax in [0, 1, 2]:
        rip[ax] = np.ceil(shape_phantom_px[ax] / shape_soul_px[ax])
    n_tot_soul = np.prod(rip)

    # create empty phantom where shape_phantom is multiple of shape_soul
    # NB - I will crop the phantom at the end of the composition
    phantom = np.zeros((rip * shape_soul_px).astype(int)).astype(esp_zero.dtype)

    # create two random distributions from the passedcsigmas with #random_angles = #souls inside the phantom
    theta_dist = np.random.normal(mu_angles[0], sigma_theta, n_tot_soul)
    phi_dist = np.random.normal(mu_angles[1], sigma_phi, n_tot_soul)

    # fill the phantom with randonmly rotated souls
    phantom = fill_with_random_rotations(phantom=phantom,
                                         esp_zero=esp_zero,
                                         shape_soul_px=shape_soul_px,
                                         ripetition_step=rip,
                                         pixel_size=np.array([param['res_xy'], param['res_xy'], param['res_z']]),
                                         theta_dist=theta_dist,
                                         phi_dist=phi_dist,
                                         border=filter_border,
                                         filter_size=filter_size)
    if _blur_inters:
        # blur in the intersection
        phantom = smooth_intersection(phantom, shape_soul_px,
                                      border=filter_border, filter_size=filter_size)

    # crop at the desidered shape
    cropped_phantom = np.copy(phantom[0: shape_phantom_px[0],
                              0: shape_phantom_px[1],
                              0: shape_phantom_px[2]])

    # if mask is passed, segment the phantom
    if mask is not None:
        return cropped_phantom * mask.astype(np.bool), theta_dist, phi_dist
    else:
        return cropped_phantom, theta_dist, phi_dist


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
    basepath = os.path.dirname(filepath)

    # ===========================================================
    # ==================    SETTINGS     =======================
    # ==========================================================

    # parameters filename
    par_filename = 'parameters.txt'

    # phantom shape (used if mask is not passed):
    shape_phantom_manual_um = np.array([500, 250, 250]) if not _test else np.array([200, 100, 40])

    # soul shape
    shape_soul_um = np.array([80, 20, 20])  # shape of 3d soul

    output_folder = 'phantoms_blur_intersections' if _blur_inters else 'phantoms_no_blur_inters'
    orientation_script_name = 'extract_orientations.py'
    results_filename = 'GAMMA_orientation_results.txt'

    # Parameters of blurring filter in the intersection (if '-b' is passed)
    # ADVICE: virtual_sample => False, real_sample => True
    filter_b = 3  # pixel selecetd on both side [-b, +b] around intersection
    filter_size = 3  # sigma of median filter

    # centers and std dev of theta and phi distribution for virtual disarray
    mu_angles = (0, 0)  # center of the angles distributions
    if _test:
        sigma_angles_range = [[10], [10]]
    else:
        sigma_angles_range = [range(0, 35, 5), range(0, 35, 5)]  # ok articolo (sigmaTHETA, sigmaPHI)
        sigma_angles_range = [[10], [10]]  # disarray al 3%
    # ===========================================================
    # ===========================================================

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
    n_phantom = len(sigma_angles_range[0]) * len(sigma_angles_range[1])
    align_mtrx = np.zeros((len(sigma_angles_range[0]), len(sigma_angles_range[1])))
    dis_area_mtrx = np.zeros((len(sigma_angles_range[0]), len(sigma_angles_range[1])))
    dis_std_mtrx = np.zeros((len(sigma_angles_range[0]), len(sigma_angles_range[1])))

    # 2.0--------------------------------------------------------------------------
    dis_area_xz_mtrx = np.zeros((len(sigma_angles_range[0]), len(sigma_angles_range[1])))
    dev_std_x_mtrx = np.zeros((len(sigma_angles_range[0]), len(sigma_angles_range[1])))
    dev_std_z_mtrx = np.zeros((len(sigma_angles_range[0]), len(sigma_angles_range[1])))
    dev_std_sumxz_mtrx = np.zeros((len(sigma_angles_range[0]), len(sigma_angles_range[1])))
    dev_std_sumxz_norm_mtrx = np.zeros((len(sigma_angles_range[0]), len(sigma_angles_range[1])))
    # -----------------------------------------------------------------------------

    print('-----------------------  -  ATTENTION  - ------------------------\n')
    print('--------   You will create {0:4.0f} panthom samples, ok?   ----------   '.format(n_phantom))
    if not _keep:
        print('------------------ All tiff files will be deleted   -------------')
    if _test:
        print('-----    Script is executing in \'TEST\' configuration   ----------')
    if _blur_inters:
        print('-----    Intersection will be smoothed with (b,s)=({},{})   ------'.format(filter_b, filter_size))
    print('-----------------------------------------------------------------)')
    print('Shape of phantom: ', shape_phantom_um, 'um ---> ', shape_phantom_px, 'px')
    print('Shape of soul   : ', shape_soul_um, 'um ---> ', shape_soul_px, 'px')

    # save elaboration time for each iteration (for estimate mean time of elaboration)
    phantom_elab_time_list = list()
    start_tot = time.time()
    count = 1

    print('\n Start creation and analysis of the series of phantom \n')

    for (r, sigma_theta) in enumerate(sigma_angles_range[0]):
        for (c, sigma_phi) in enumerate(sigma_angles_range[1]):

            print('  - n\' {} of {} - Creating and analyzing with sigma(theta, phi) = ({},{})...'.format(
                count, n_phantom, sigma_theta, sigma_phi), end='')

            elab_start = time.time()

            phantom, theta_dist, phi_dist = create_phantom_disarray(esp_zero=esp_zero,
                                                                    param=param,
                                                                    sigma_theta=sigma_theta,
                                                                    sigma_phi=sigma_phi,
                                                                    mu_angles=mu_angles,
                                                                    shape_soul_px=shape_soul_px,
                                                                    shape_phantom_px=shape_phantom_px,
                                                                    mask=mask,
                                                                    _blur_inters=_blur_inters,
                                                                    filter_border=filter_b,
                                                                    filter_size=filter_size)

            # define current paths
            sub_folder = 'analyze_phantom_sigma_{0:0.0f}_{1:0.0f}'.format(sigma_theta, sigma_phi)
            seq_folder = 'segmented/stack'
            current_path = os.path.join(basepath, output_folder, sub_folder)

            # create current phantom folder
            if not os.path.isdir(os.path.join(current_path, seq_folder)):
                os.makedirs(os.path.join(current_path, seq_folder))

            # save current distribution
            np.save(os.path.join(current_path, 'theta_dist.npy'), theta_dist)
            np.save(os.path.join(current_path, 'phi_dist.npy'), phi_dist)

            # save as image sequence of tiff in the folder
            for z in range(phantom.shape[2]):
                img_z_name = create_img_name_from_index(z, pre='phantom_')
                tifffile.imsave(
                    os.path.join(current_path, seq_folder, img_z_name), phantom[..., z])

            # copy parameters.txt file
            shutil.copyfile(par_filepath, os.path.join(current_path, par_filename))

            # analyze the sample with external SCRIPT and write results and output in two txt
            os.system('python3 {} -sf {} > {}'.format(
                orientation_script_fullpath,
                os.path.join(current_path, seq_folder),
                os.path.join(current_path, 'out_execution.txt')))

            # read current results from .txt
            results_txtpath = os.path.join(current_path, results_filename)
            words = all_words_in_txt(results_txtpath)

            # write the results in the matrices
            align_mtrx[r, c] = words[words.index('Alignment') + 2]
            dis_area_mtrx[r, c] = words[words.index('(area_ratio)') + 2]
            dis_std_mtrx[r, c] = words[words.index('dev.std)') + 2]

            # 1.1 --------------------------
            dis_area_xz_mtrx[r, c] = words[words.index('2.0_area_ratio_xz') + 2]
            dev_std_x_mtrx[r, c] = words[words.index('2.0_std_dev_x') + 2]
            dev_std_z_mtrx[r, c] = words[words.index('2.0_std_dev_z') + 2]
            dev_std_sumxz_mtrx[r, c] = words[words.index('2.0_sum_std_xz') + 2]
            dev_std_sumxz_norm_mtrx[r, c] = words[words.index('2.0_sum_std_xz_norm') + 2]
            # ---------------------------------------

            # 2.0 --------------------------
            # di fatto serve solo salvare R,
            # poi con lo script Jupyter mi estraggo tutti i disarray locali da R e ne salvo la media
            # per ogni phantom creato
            #àà-----------------------------

            if not _keep:
                # delete tiff files of current sample
                shutil.rmtree(os.path.join(current_path, seq_folder))

            phantom_elab_time_list.append(time.time() - elab_start)
            elapsed_time = (n_phantom - count) * np.mean(phantom_elab_time_list)
            (h, m, s) = seconds_to_hour_min_sec(elapsed_time)

            count = count + 1
            print(' ---   ET: {0:2d}h {1:2d}m {2:2d}s'.format(int(h), int(m), int(s)))

    # savings
    np.save(os.path.join(basepath, 'sigma_theta_tested.npy'), np.array(sigma_angles_range[0]))
    np.save(os.path.join(basepath, 'sigma_phi_tested.npy'), np.array(sigma_angles_range[1]))
    np.save(os.path.join(basepath, 'align_mtrx.npy'), align_mtrx)
    np.save(os.path.join(basepath, 'dis_area_mtrx.npy'), dis_area_mtrx)
    np.save(os.path.join(basepath, 'dis_std_mtrx.npy'), dis_std_mtrx)
    # 2.0:
    np.save(os.path.join(basepath, 'dis_area_xz_mtrx.npy'), dis_area_xz_mtrx)
    np.save(os.path.join(basepath, 'dev_std_x_mtrx.npy'), dev_std_x_mtrx)
    np.save(os.path.join(basepath, 'dev_std_z_mtrx.npy'), dev_std_z_mtrx)
    np.save(os.path.join(basepath, 'dev_std_sumxz_mtrx.npy'), dev_std_sumxz_mtrx)
    np.save(os.path.join(basepath, 'dev_std_sumxz_norm_mtrx.npy'), dev_std_sumxz_norm_mtrx)

    end_tot = time.time()
    print_time_execution(start=start_tot, end=end_tot, proc_name='\nCreated {} phantom - '.format(n_phantom))

    return None


if __name__ == '__main__':
    main()
