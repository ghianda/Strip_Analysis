import numpy as np
import os
from PIL import Image
from io import BytesIO

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse

import scipy.ndimage as ndimage
from skimage import exposure
from skimage import img_as_float
from tifffile import imsave

from custom_tool_kit import magnitude


class ImgFrmt:
    # image_format STRINGS
    EPS = 'EPS'
    TIFF = 'TIFF'
    SVG = 'SVG'


def plot_map_and_save(matrix, np_filename, base_path, shape_G, shape_P, img_format=ImgFrmt.TIFF, _do_norm=False):
    """
    # plot LOCAL disarray (or AVERAGED FA) matrix as frames

    # map_name = 'FA', or 'DISARRAY_ARIT' or 'DISARRAY_WEIGH'
    # es: plot_map_and_save(matrix_of_disarray, disarray_numpy_filename, True, IMG_TIFF)
    # es: plot_map_and_save(matrix_of_local_FA, FA_numpy_filename, True, IMG_TIFF)

    :param matrix:
    :param np_filename:
    :param save_plot:
    :param img_format:
    :param _do_norm:
    :return:
    """

    # create folder_path and filename from numpy_filename
    plot_folder_name = np_filename.split('.')[0]
    plot_filebasename = '_'.join(np_filename.split('.')[0].split('_')[0:2])

    # create path where save images
    plot_path = os.path.join(base_path, plot_folder_name)
    # check if it exist
    if not os.path.isdir(plot_path):
        os.mkdir(plot_path)

    # iteration on the z axis
    for i in range(0, matrix.shape[2]):

        # extract data from the frame to plot
        if _do_norm:
            img = normalize(matrix[..., i])
        else:
            img = matrix[..., i]

        # evaluate the depth in the volume space
        z_frame = int((i + 0.5) * shape_G[2] * shape_P[2])
        # create title of figure
        title = plot_filebasename + '. Grane: ({} x {} x {}) vectors; Depth_in_frame = {}'.format(
            int(shape_G[0]), int(shape_G[1]), int(shape_G[2]), z_frame)

        # create plot
        fig = plt.figure(figsize=(15, 15))
        plt.imshow(img, cmap='gray')
        plt.title(title)
        # plt.show()

        # create fname for this frame
        fname = plot_filebasename + '_z={}'.format(z_frame)

        if img_format == ImgFrmt.SVG:
            # formato SVG -> puoi decidere dopo la risoluzione aprendolo con fiji
            fig.savefig(str(os.path.join(plot_path, fname) + '.svg'), format='svg',
                        dpi=1200, bbox_inches='tight', pad_inches=0)

        elif img_format == ImgFrmt.EPS:
            # formato EPS buono per latex (latex lo converte automat. in pdf)
            fig.savefig(str(os.path.join(plot_path, fname) + '_black.eps'), format='eps', dpi=400,
                        bbox_inches='tight', pad_inches=0)

        elif img_format == ImgFrmt.TIFF:
            png1 = BytesIO()
            fig.savefig(png1, format='png')
            png2 = Image.open(png1)
            png2.save((str(os.path.join(plot_path, fname) + '.tiff')))
            png1.close()

        plt.close(fig)

    return plot_path


def save_tiff(img, img_name, comment='', folder_path='', prefix=''):
    # folder_path must end with '/'

    # check if name end with .tif
    if img_name.endswith('.tif'):
        base, ext = os.path.splitext(img_name)
        img_name = prefix + base + '_' + comment + '.tif'
    else:
        img_name = prefix + img_name + '_' + comment + '.tif'

    # check if path end with '/'
    if not folder_path.endswith('/'):
        folder_path += '/'

    imsave(folder_path + img_name, (img * (255 / np.max(img))).astype(np.uint8))


def save_eps(data, filename, to_folder_path, comment='', prefix='', cmap='gray', dpi=300):
    "data is a numpy.ndarray or a matplolib.pyplot module"
    #to_folder_path must end with '/'
    # plt.ioff()
    if data is not None:
        if type(data) is np.ndarray:
            f = plt.figure()
            plt.imshow((data * (255 / np.max(data))).astype(np.uint8), cmap=cmap)
            plt.savefig(to_folder_path + prefix + filename + '_' + comment + '.eps', format='eps', dpi=dpi)
            plt.close(f)
        else:
            #data is a matplotlib.pyplot module
            data.savefig(to_folder_path + prefix + filename + '_' + comment + '.eps', format='eps', dpi=dpi)
    else: print(' - ERROR (save_eps): data is None. No save .eps')
    # inserire in savefig() bbox_inches='tight'
    
    
def save_svg(fig, plt, output_folder_path, filename, bbox_inches='tight', pad_inches=0, dpi=1200):
    # save the plot in a svg image inside output_folder_path with name: filename.save_svg
    # if fig and plt are not None, save fig and return True, else return False
    if fig is not None and plt is not None:                    
        img_name = str(filename + '.svg')
        dest_path = os.path.join(output_folder_path, img_name)
        fig.savefig(dest_path, format='svg', dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
        plt.close(fig)
        return True
    else:
        return False


def image_loader(file_name, folder_path, channel=0):
    if not file_name.endswith('.tif'):
        file_name += '.tif'

    file_path = folder_path + file_name
    img = Image.open(file_path)
    if img.mode == 'I;16B':
        X = np.array(img)//257
    elif img.mode == 'L' or img.mode == 'P':
        X = np.array(img)
    elif img.mode == 'RGB':
        X = np.array(img)[:, :, channel]
    else:
        raise ValueError('Dunno how to handle image mode %s' % img.mode)
    return X


def image_have_info(img, t=0.09):
    rating = np.sum(img) / np.prod(img.shape)  
    if rating > t :
        return True
    else:
        return False


def normalize(img, max_value=255.0):
    max_v = img.max()
    min_v = img.min()
    if max_v != 0:
        if max_v != min_v :
            return (((img - min_v)/(max_v - min_v)) * max_value).astype(np.uint8)
        else:
            return ((img / max_v) * max_value).astype(np.uint8)
    else:
        return (img).astype(np.uint8)


# def normalize(img ,max_value=255.0):
#     if max_value > 1 :
#         return ((img - img.min())/(img.max() - img.min()) * max_value).astype(np.uint8)
#     elif max_value == 1 :
#         return ((img - img.min())/(img.max() - img.min()) * max_value).astype(np.float32)
#     else: 
#         print('max_value must be 1 or greater then 1 -> insert: {} -> return None'.format(max_value))
#         return None



# def normalize(img, type='uint8'):
#     norm = 255.*(img - np.min(img)) / np.maximum((np.max(img) - np.min(img)), 1.)
#     return {
#         'uint8': norm.astype(np.uint8),
#         'float16': norm.astype(np.float16),
#         'float32': norm.astype(np.float32),
#         'float64': norm.astype(np.float64)
#     }.get(type, norm.astype(np.uint8))  # uint8 is default dtype


def split_channel(img):
    img_red = np.array(img)[:, :, 0]
    img_green = np.array(img)[:, :, 1]
    img_blue = np.array(img)[:, :, 2]
    return img_red, img_green, img_blue


def print_image_info(img, title=''):
    if img is None:
        return None
    print(title)
    print(' * Image mode: ', img.mode)
    print(' * Image shape: ', np.array(img).shape)
    print(' * Image dtype: {}'.format(img.dtype))
    
    
def print_info(X, text=''):
    if X is None:
        return None
    print(text)
    print(' * Image dtype: {}'.format(X.dtype))
    print(' * Image shape: {}'.format(X.shape))
    print(' * Image max value: {}'.format(X.max()))
    print(' * Image min value: {}'.format(X.min()))
    print(' * Image mean value: {}'.format(X.mean()))
    
    
# plot projection of external face of the cube (assial, coronal and sagittal)
def plot_external_projection(X, _return=False, figsize=(30, 10), global_title=''):
    # X : numpy.ndarray

    # old:------------------------------------------
    # shape = X.shape
    # side_yz = np.flipud(np.rot90(X[:,shape[1]-1,:]))
    # side_xz = np.flipud(np.rot90(X[shape[0]-1,:,:]))
    # side_xy = X[:, :, 0]

    # plot_n_subplots(
    #     img_list=(side_xy, side_yz, side_xz),
    #     sub_titles_array=['XY', 'YZ', 'XZ'],
    #     global_title=global_title + ' - Projection of external face',
    #     figsize=figsize,
    #     fontsize_glbltit=30,
    #     fontsize_subtit=30)
    #
    # if _return:
    #     return side_xy, side_xz, side_yz
    #-------------------------------------------------

    # NEW (YXZ):
    side_yx = X[..., 0]  # first plane yx
    side_zx = np.flipud(np.rot90(X[-1, ...]))  # frontal face (bottom)
    side_yz = np.flipud(np.rot90(X[:, -1, :]))  # side face (right)

    plot_n_subplots(
        img_list=(side_yx, side_zx, side_yz),
        sub_titles_array=['YX (Z=0)', 'ZX (Y=end)', 'YZ (X=end)'],
        global_title=global_title + ' - Projection of external faces',
        figsize=figsize,
        fontsize_glbltit=30,
        fontsize_subtit=30)

    if _return:
        return side_yx, side_zx, side_yz


def plot_central_projection(X, _return=False, figsize=(30, 10), global_title=''):
    # X : numpy.ndarray
    s = X.shape

    # xcentral cordinates:
    y, x, z = int(s[0]/2), int(s[1]/2), int(s[2]/2)
    side_yx = X[..., z]
    side_zx = np.flipud(np.rot90(X[y, ...]))
    side_yz = np.flipud(np.rot90(X[:, x, :]))

    plot_n_subplots(
        img_list=(side_yx, side_zx, side_yz),
        sub_titles_array=['YX (Z={})'.format(z), 'ZX (Y={})'.format(y), 'YZ (X={})'.format(x)],
        global_title=global_title + ' - Projection of central planes',
        figsize=figsize,
        fontsize_glbltit=30,
        fontsize_subtit=30)

    if _return:
        return side_yx, side_zx, side_yz


def plot_histogram(img, title='', bins=256):
    # plot histogram
    plt.figure()
    plt.hist(img.flatten(), bins=bins, fc='k', ec='k')
    plt.title(title)
    # plot cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(img, bins)
    plt.plot(bins, img_cdf, 'r')
    return plt



def divide_images(img1,img2):
    if img1 is not None and img2 is not None:
        if 0 in img2:
            img2[img2 == 0] = 1
        img1_f = img_as_float(img1)
        img2_f = img_as_float(img2)
        return img1_f / img2_f
    else:
        return None

    
def low_pass_filter(img):
    img_filtered = ndimage.gaussian_filter(img, sigma=5, order=0)
    return img_filtered


def plot_image(X, title='', _grid=False, spacing=1, size=(14, 14), _gray=True):
    fig = plt.figure(figsize=size)
    plt.title(title)
    
    if _gray:
        plt.imshow(X, cmap='gray')
    else:
        plt.imshow(X)
    
    #if _grid:
        # DEPRECATED
        # minorLocator = MultipleLocator(spacing)
        # # Set minor tick locations.
        # ax = fig.gca()
        # ax.yaxis.set_minor_locator(minorLocator)
        # ax.xaxis.set_minor_locator(minorLocator)
        # # Set grid to use minor tick locations.
        # plt.grid(which='minor')

    # TODO da sistemare unità di misura
    # if scale is not None:
        # scalebar = ScaleBar(scale) # 1 pixel = (scale) micro-meter, es: scale = 200
        # plt.gca().add_artist(scalebar)

    plt.show(block=False)
    return plt


def plot_couple_img_hist(X, title='', cmap='gray'):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,7))
    f.suptitle(title, fontsize=14)
    ax1.imshow(X, cmap=cmap)
    ax2.hist(X.flatten(), bins=256, fc='k', ec='k')
    plt.show(block=False)
    return plt


def plot_n_subplots(img_list, sub_titles_array, global_title='', cmap='gray', _plot=True, fontsize_glbltit=25, fontsize_subtit=20, figsize=(30, 10), _axs=True):

    if _plot is False: plt.ioff()   # Turn interactive plotting off

    n = len(img_list)
    if n < 0 or n > 10:
        print('ZERO OR TOO MANY IMAGES FOR SUBPLOTTING -> MODIFY CODE')
        return None
    if 0 < n <= 5:
        f, axs = plt.subplots(1, n, figsize=figsize)
        f.suptitle(global_title, fontsize=fontsize_glbltit)
        for i in range(0,n):
            axs[i].imshow(img_list[i], cmap=cmap)
            axs[i].set_title(sub_titles_array[i], fontsize=fontsize_subtit)
            if ~_axs: axs[i].axis('off')
        plt.subplots_adjust(left=0.01, bottom=0.03, right=0.99, top=0.9, wspace=0.05, hspace=0.20)
        return plt, f
    if 5 < n <= 10:
        f, axs = plt.subplots(2, 5, figsize=figsize)
        f.suptitle(global_title, fontsize=fontsize_glbltit)
        for row in range(0,2):
            for i in range(0,5):
                axs[row,i].imshow(img_list[row*5 + i], cmap=cmap)
                axs[row,i].set_title(sub_titles_array[row*5 + i], fontsize=fontsize_subtit)
                if ~_axs: axs[row,i].axis('off')
        plt.subplots_adjust(left=0.01, bottom=0.03, right=0.99, top=0.9, wspace=0.05, hspace=0.20)
        return plt, f


def plot_3d(Z, title=''):
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    x, y = Z.shape
    X = np.arange(0, x, 1)
    Y = np.arange(0, y, 1)
    X, Y = np.meshgrid(X, Y)
    
    # delete inf values
    Z[Z == -np.inf] = 0

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(Z.min()-1, Z.max()+1)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    # axes labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title(title)
    plt.show(block=False)


def create_img_name_from_index(i, pre='', post='', tot_digits=5):
    if 0 <= i < 10 ** tot_digits - 1:
        if i == 0:
            non_zero_digits = 1
        else:
            non_zero_digits = magnitude(i)+1
        zeros = '0' * (tot_digits - non_zero_digits)
        return str(pre + zeros + str(i) + post + '.tif')
    else:
        return 'image_index_out_of_range'



# funzione che plotta in modo interattivo un ndarray numpy
# dependencies:
# numpy as np
# import nibabel as nib
# from niwidgets import NiftiWidget
# import nilearn.plotting as nip
# def interactive_plot(X, affine=None, pro=False, rot=False):
#     import nibabel as nib
#     from niwidget_volume import NiftiWidget

#     data = np.copy(X)
#     shape = data.shape
#     if len(shape) == 3:
#         if affine is None:
#             affine = np.eye(4)
#         if rot:
#             # rot 90' clockwise (plane xy)
#             data = np.rot90(data, k=3)
#         data = normalize(data)
#         data_nii = nib.Nifti1Image(data, affine)
#         widget = NiftiWidget(data_nii)
#         if pro:
#             widget.nifti_plotter(
#                 plotting_func=nip.plot_epi, 
#                 display_mode=['ortho', 'x', 'y', 'z', 'yx', 'xz', 'yz'],
#                 threshold=(0.0, 255.0, 1.0)
#                 )
#         else:
#             widget.nifti_plotter()
#         return data_nii, widget
#     else:
#         print('Error: X have {} axes'.format(len(shape)))
#         return None
    

def turn_in_niwidget_coords(coords, shape):
    (row, column, slices) = tuple(shape)
    (x, y, z) = tuple(coords)
    
    x_nii = y
    y_nii = (row - 1) - x
    z_nii = z
    
    return (x_nii, y_nii, z_nii)


# plot projection of selected function( max, min or mean)
def plot_projection(X, func='max', _return=False, figsize=(30, 10), global_title=''):
    # X : numpy.ndarray
    
    #Double-check: only allowed methods and X must have it!
    allowed_commands = ['max', 'mean', 'min']
    assert func in allowed_commands, "Command '%s' is not allowed"%func

    func_on_X = getattr(X, func, None)
    assert callable(func_on_X), "Command '%s' is invalid"%func
    
       
    proiez_xz = np.flipud(np.rot90(func_on_X(axis=0)))
    proiez_yz = np.flipud(np.rot90(func_on_X(axis=1)))
    proiez_xy = func_on_X(axis=2)
    
    plot_n_subplots(
        img_list = (proiez_xy, proiez_xz, proiez_yz), 
        sub_titles_array = ['XY', 'XZ', 'YZ'], 
        global_title = global_title + ' - Projection of numpy.{}()'.format(func),
        figsize=figsize,
        fontsize_glbltit=10,
        fontsize_subtit=10)
    
    if _return:
        return proiez_xy, proiez_xz, proiez_yz


def scatter_plot_3D(points, axes_lim, title='', units=['', '', ''],
                        points_size=10, _centroid=False, centroid_size=10, 
                        center=None, subspace=None, _return=False): 
    # points: list of tuple(r,c,z), every tuple is a 3d point
    # axes_lim: dimension of cube where plot points
    # - if len(axes_lim) == 3 -> (xmax, ymax, zmax)
    # - if len(axes_lim) == 6 -> (xmin, xmax, ymin, ymax, zmin, zmax)
    # _centroid (bool) : if True, estimate centroid of points and plots it
    # units : list of three string (units of axes)
    # center : tuple of center coordinate {optional}
    # subspace : {integer} - [accepeted values: 0,1,2]
    # - if passed, this refer to axes that define seleceted subspace:
    # - - subspace == 0 -> all points are moved to X>0 subspace
    # - - subspace == 1 -> all points are moved to Y>0 subspace
    # - - subspace == 2 -> all points are moved to Z>0 subspace

    if type(points) is not list:
        points = list(points)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # plt.axis('scaled')  dava errore
    
    # put points in the right subspace
    # (with selected coord > 0)
    # subspace = 0 -> points with x > 0
    # subspace = 1 -> points with y > 0
    # subspace = 2 -> points with z > 0
    if subspace is not None and subspace in [0,1,2]:
        # in scatter plot Y and X are inverted in Image Standard System
        if subspace == 0:
            mirror_ax = 1
        elif subspace == 1:
            mirror_ax = 0
        else:
            mirror_ax = 2
        
        # move points in slelected subspace
        points_to_plot = []
        for p in points:
            # check values on mirror axis:
            if p[mirror_ax] < 0:
                points_to_plot.append((-p[0], -p[1], -p[2]))
            else:
                points_to_plot.append((p[0], p[1], p[2]))
    else:
        points_to_plot = points
        
    # extract coordinates from list of points
    X = [coord[0] for coord in points_to_plot]
    Y = [coord[1] for coord in points_to_plot]
    Z = [coord[2] for coord in points_to_plot]

    ax.set_xlabel('x ' + units[0])
    ax.set_ylabel('y ' + units[1])
    ax.set_zlabel('z ' + units[2])
    plt.title(title)

    if len(axes_lim) == 3:
        ax.set_xlim(0, axes_lim[1])
        ax.set_ylim(0, axes_lim[0])
        ax.set_zlim(0, axes_lim[2])
    elif len(axes_lim) == 6:
        ax.set_xlim((axes_lim[2], axes_lim[3]))
        ax.set_ylim((axes_lim[0], axes_lim[1]))
        ax.set_zlim((axes_lim[4], axes_lim[5]))

    # NB - in scatter plot, axes are (y,x,z) in image standard system
    ax.scatter(Y, X, Z, s=points_size)
    
    if center is not None:
        ax.scatter(center[1], center[0], center[2], color="#FF0080", s=points_size*100)
    
    if _centroid:
        x_cen = np.mean(X)
        y_cen = np.mean(Y)
        z_cen = np.mean(Z)
        ax.scatter(y_cen, x_cen, z_cen, color="#FF0080", s=centroid_size)
    
    
    plt.show()
    if _return:
        return ax, points_to_plot


def scatter_plot_2D_angular_dispersion(points, axes_lim, title='', axes_name=('',''), 
                    points_size=10, _centroid=False, centroid_size=10, _circle=True, radius=0, 
                    _ellipsis=False, dev_0=None, dev_1=None, z_limit=None, z_dashes=None):
    # points: list of tuple(r,c), every tuple is a 2d point
    # shape: dimension of cube where plot points
    # _centroid : if True, estimate centroid of points and plots it
    # _circle : if True, draw red circle with radius = radius
    # _ellipsis : if True, draw error ellipsis (dispersion of data).
    # Estimate standard deviation of 1th and 2th component of data,
    # and draw an ellipsis with:
    # - center : centroid of data
    # - 1th axis : dev. standard of 1th components of data
    # - 2th axis : dev. standard of 2th components of data
    # - Ellipsis use dev_0 and dev_1 if passed, otherwise estimate it.
    # z_limit = value of limit in 2th axis. If passed, plot horizontal line on +z_limit and -z_limit
    
    if type(points) is not list:
        points = list(points)
    
    fig = plt.figure()
    ax = plt.gca()
    plt.axis('scaled')
    
    # extract coordinates 
    X0 = [coord[0] for coord in points]
    X1 = [coord[1] for coord in points]
    
    if _centroid or _ellipsis:
        # calculate centroid cordinates
        x0_cen = np.mean(X0)
        x1_cen = np.mean(X1)

    # title and axes names
    plt.xlabel(axes_name[0])
    plt.ylabel(axes_name[1])
    plt.title(title)

    # define axes limits
    if len(axes_lim) == 2:
        inf0 = inf1 = 0
        sup0 = axes_lim[0]
        sup1 = axes_lim[1]
    else:
        inf0 = axes_lim[0]
        sup0 = axes_lim[1]
        inf1 = axes_lim[2]
        sup1 = axes_lim[3]
    
    # set axes limits
    plt.xlim(inf0, sup0)
    plt.ylim(inf1, sup1)
    
    # plot points
    plt.scatter(X0, X1, s=points_size)
    # plt.hexbin(X,Y)  # istogramma brutto
    
    # plot centroids
    if _centroid:
        plt.scatter(x0_cen, x1_cen, color="#FF0080", s=centroid_size)
    
    # draw red circle
    if _circle:
        radius = radius
        circle = plt.Circle((0, 0), radius, color='r', fill=False)
        ax.add_artist(circle)
        
        # add ticks on radius
#         plt.xticks(list(plt.xticks()[0]) + [radius])
#         plt.yticks(list(plt.yticks()[0]) + [radius])
        # add radius line
        
    # draw ellipsis of standard deviation
    if _ellipsis:
        if dev_0 is None or dev_1 is None:
            arr = np.array(points)
            dev_0 = np.std(arr[:,0])
            dev_1 = np.std(arr[:,1])
            
        ell = Ellipse(xy=(x0_cen, x1_cen), width=2*dev_0, height=2*dev_1,
                      fill=False, color='r')
        ax.add_artist(ell)
        
        # plot axes of ellisis
        plt.plot([x0_cen-dev_0, x0_cen+dev_0], [x1_cen, x1_cen], 'r', lw=0.5)
        plt.plot([x0_cen, x0_cen], [x1_cen-dev_1, x1_cen+dev_1], 'r', lw=0.5)
    
    if z_limit is not None:
        # draw z limit
        plt.plot((inf0, sup0), (z_limit, z_limit), 'k', lw=0.5, linestyle='--', dashes=z_dashes)
        plt.plot((inf0, sup0), (-z_limit, -z_limit), 'k', lw=0.5, linestyle='--', dashes=z_dashes)
        
        # draw outside area
#         wedge = mpatches.Wedge(center=(0,z_limit), )
#         .Wedge(grid[2], 0.1, 30, 270, ec="none")

    #spine placement data centered
    ax.spines['left'].set_position(('data', 0.0))
    ax.spines['bottom'].set_position(('data', 0.0))
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
        
    # show figure
    plt.show()


def scatter_plot_2D(points, shape, title='', axes_name=('x','y'), points_size=10, _image_std=True):
    # points: list of tuple(r,c), every tuple is a 2d point
    # shape: dimension of cube where plot points
    #
    if type(points) is not list:
        points = list(points)
    
    fig = plt.figure()
    ax = plt.gca()
    plt.axis('scaled')
       
    # extract coordinates 
    X = [coord[0] for coord in points]
    Y = [coord[1] for coord in points]
    
    # calculate centroid cordinates
    x_cen = np.mean(X)
    y_cen = np.mean(Y)

    # title and axes names
    plt.xlabel(axes_name[0])
    plt.ylabel(axes_name[1])
    plt.title(title)

    # axes lim     
    if _image_std:
        plt.xlim((shape[2], shape[3]))
        plt.ylim((shape[0], shape[1]))
        
        # reverse y axis (Image standard system)
        plt.gca().invert_yaxis()
        
        # plot points
        plt.scatter(Y, X, s=points_size)
    
    else:
        plt.xlim((shape[0], shape[1]))
        plt.ylim((shape[2], shape[3]))
        
        # plot points
        plt.scatter(X, Y, s=points_size)
     
        
    # show figure
    plt.show()


def plot_quiver_3d_entire_volume(x0_c, x1_c, x2_c, x0_q, x1_q, x2_q, axes_label=('','',''), shape=None, title='', scale=None, size=None, color='r',
                  _fiber_axis=False, fiber_lw=20):
    # xc, yc, zc -> array of coordinates of TAIL of the Arrow
    # xq, yq, zq -> array of components of quiver (Head of artow relative to tail)

    fig = plt.figure(figsize=size)
    ax = fig.gca(projection='3d')
    
    plt.title(title)
    
    if axes_label is not None:
        ax.set_xlabel(axes_label[0])
        ax.set_ylabel(axes_label[1])
        ax.set_zlabel(axes_label[2])

    
    ax.set_xticks([0,100,200,300,400])
    ax.set_yticks([0,500,1000,1500,2000])
    ax.set_zticks([0,100,200,300,400])
    
    # ax.zticks
    
    if shape is not None:
        ax.set_xlim((0, shape[0]))
        ax.set_ylim((0, shape[1]))
        ax.set_zlim((0, shape[2]))
        
    # scale axes proportion by dilation matrix coefficient:
    if scale is not None:
        ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax),
                                 np.diag([scale[0], scale[1], scale[2], 1]))

    # real peaks quiver
    # ax.quiver(x0_c, x1_c, x2_c, x0_q, x1_q, x2_q)

    # bigger quivers for see results 
    ax.quiver(x0_c, x1_c, x2_c,
              x0_q, x1_q, x2_q,
              length=10,
              normalize=True,
              arrow_length_ratio=0,
              pivot='middle',
              color=color)
    
    if _fiber_axis:
        # y axial quiver
        ax.quiver(200, 0, 150,
                  0, shape[1], 0,
                  lw=fiber_lw)
    
    plt.show()


def scatter_hist_2d(points, shape, title='', axes_name=('x','y'), 
                    points_size=10, _centroid=False, _circle=True, radius=0):
    # points: list of tuple(r,c), every tuple is a 2d point
    # shape: dimension of cube where plot points
    #
    if type(points) is not list:
        points = list(points)
    
    fig = plt.figure()
    
    # extract coordinates 
    X = [coord[0] for coord in points]
    Y = [coord[1] for coord in points]

    # title and axes names
    plt.xlabel(axes_name[0])
    plt.ylabel(axes_name[1])
    plt.title(title)

    # axes lim 
    plt.xlim((shape[0], shape[1]))
    plt.ylim((shape[2], shape[3]))

    xedges = np.linspace(shape[0], shape[1], 42)
    yedges = np.linspace(shape[2], shape[3], 42)
    hist, xedges, yedges = np.histogram2d(X, Y, (xedges, yedges))
    xidx = np.clip(np.digitize(X, xedges), 0, hist.shape[0]-1)
    yidx = np.clip(np.digitize(Y, yedges), 0, hist.shape[1]-1)
    c = hist[xidx, yidx]
    plt.scatter(X, Y, c=c, s=0.1)
    
    # plot centroids
    if _centroid:
        x_cen = np.mean(X)
        y_cen = np.mean(Y)
        plt.scatter(x_cen, y_cen, color="#FF0080", s=100 * points_size)
    
    # draw red circle
    if _circle:
        # patches = []
        # resolution = 50
        x0 = 0
        y0 = 0
        radius = radius
        # circle = Circle((x0, y0), radius, color='r', fill=False)
        circle = plt.Circle((x0, y0), radius, color='r', fill=False)
        # patches.append(circle)
        # p = PatchCollection(patches)
        # ax.add_collection(p)
        ax = plt.gca()
        ax.add_artist(circle)
        
    # show figure
    plt.show()


def plot_quiver_3d(x0_c, x1_c, x2_c, x0_q, x1_q, x2_q, axis_label=('','',''), shape=None, title='', size=None, color='b'):
    # xc, yc, zc -> array of coordinates of TAIL of the Arrow
    # xq, yq, zq -> array of components of quiver (Head of artow relative to tail)

    fig = plt.figure(figsize=size)
    ax = fig.gca(projection='3d')
    
    plt.title(title)
    
    ax.set_xlabel(axis_label[0])
    ax.set_ylabel(axis_label[1])
    ax.set_zlabel(axis_label[2])
    
    if shape is not None:
        ax.set_xlim((0, shape[0]))
        ax.set_ylim((0, shape[1]))
        ax.set_zlim((0, shape[2]))

    # real peaks quiver
    ax.quiver(x0_c, x1_c, x2_c, x0_q, x1_q, x2_q)
    
    plt.show()


def plot_quiver_2d(x0_c, x1_c, x0_q, x1_q, axes_label=('',''), img=None, shape=None, origin='upper', title='',
                   color='r', units='xy', scale_units='xy', scale=1., pivot='middle', _img_standard=True, _return=False):
    # xc, yc -> array of coordinates of TAIL of the Arrow
    # xq, yq -> array of components of quiver (Head of artow relative to tail)

    fig = plt.figure()
    ax = fig.gca()
    ax.set_aspect('equal')
    
    if img is not None:
        plt.imshow(img, origin=origin)
        _img_standard = True
    
    
    # plot all quivers
    # plt.quiver(xc, yc, xq, yq, headlength=0, headwidth=1)
    plt.quiver(x0_c, x1_c, x0_q, x1_q,
               units=units, color=color,
               headwidth=1, headlength=0,
               scale_units=scale_units, scale=scale, pivot=pivot)
        

    plt.xlabel(axes_label[0])
    plt.ylabel(axes_label[1])
    plt.title(title)
    
    if shape is not None:
        ax.set_xlim((0, shape[0]))
        ax.set_ylim((0, shape[1]))
        
    if _img_standard:
        ax.invert_yaxis()
        
    plt.show()
    
    if _return:
        return fig
