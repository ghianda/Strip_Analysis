import glob
import os
import numpy as np
from natsort import natsorted

from skimage.external import tifffile as tif
from PIL import Image


def manage_path_argument(source_path):
    '''
    # manage input parameters:
    # if script is call by terminal, source_path is a list with one string inside (correct source path)
    # if script is call by another script (structural_analysis, for example), and if source_path contains some ' ' whitesace,
    # system split the string in the list, so it joins it

    :param source_path : variables from args.source_folder
                         it's a list with inside the path of the images to processing.
    :return 
    '''
    if type(source_path) is list:
        if len(source_path) > 1:
            # there are white spaces, system split in more string (wrong)
            given_path = ' '.join(source_path)
        else:
            given_path = source_path[0]
    else:
        given_path = source_path

    # # correct whitespace with backslash
    # given_path = given_path.replace(' ', '\ ')

    # extract base path
    if given_path.endswith('/'):
        given_path = given_path[0:-1]

    return given_path


def load_tif_data(source_path):
    '''
    Load tif file(s) in source_path into a numpy.ndarray.
    Load either from single 3D tif file or from a directory of multipl 2d tiff frame.

    :param  - source_path: path of 3d file or of directory  
    :return: numpy.ndarray of 3d data loaded
    '''

    # if source_path is directory
    if os.path.isdir(source_path):
        print(' - Loading from directory:')
        print(os.path.dirname(source_path), '\n')

        # create a sorted list of filenames
        flist = []
        flist += glob.glob(os.path.join(source_path, '*.tif*'))
        flist += glob.glob(os.path.join(source_path, '*.TIF*'))
        flist = sorted(flist)

        print('*************  -  ', len(flist))

        # open first image (or 3d tiff file) for check frame dimension
        # (Huygens save like (1, 1, row, col) while ImageJ like (row, col))
        img = tif.imread(flist[0])
        _to_reshape = False
        if len(img.shape) >= 4:
            img = img[0, 0, :, :]
            _to_reshape = True

        if len(flist) == 1:
            print('************  -  A')
            # there is a single frame or a 3d tiff file
            volume = np.copy(img)
            # if it's a 3d tiff file, axes is z,y,x -> I want y,x,z
            print(volume.shape)
            volume = np.moveaxis(volume, 0, -1)
            print(volume.shape)

        elif len(flist) > 1:
            # there is a list of 2d frames

            # create empty 'volume' (uint8 ndarray)
            volume = np.zeros((img.shape[0], img.shape[1], len(flist)), dtype=img.dtype)

            # write first image inside final volume
            volume[:, :, 0] = img

            # read all images and add to volume
            for (z, f) in enumerate(flist):
                img = tif.imread(f)
                if _to_reshape:
                    img = img[0, 0, :, :]
                volume[:, :, z] = img
                print('Loaded ', os.path.basename(f))

        volume = np.squeeze(volume)  # remove axis if dimension is 1 (es: shape[1024, 1024, 1] -> shape[1024, 1024])

    # if source_path is a .tiff filename
    else:        
        # open tif file
        volume = tif.imread(source_path)

        if len(volume.shape) == 2:
            print(' - Loaded 2D tiff file: {}'.format(os.path.basename(source_path)))

        else:
            volume = np.squeeze(volume.swapaxes(0, 2).swapaxes(0, 1))  # Swap axes from [z, r, c] (tif format) to [r, c, z]
            print(' - Loaded 3D tiff file')

        print('   Locate in: \n {}'.format(os.path.dirname(source_path)))  
    return volume


def load_stack_into_numpy_ndarray(source_path):
    ''' 
    :param source_path : list with one element: path of images
    '''
    
    # read all images in path and insert in a list
    source_data = make_dataset(source_path)
    data_length = len(source_data)

    # delete source_data items and move data inside volume
    volume = create_stack_light(source_data)
    del source_data

    if data_length == volume.shape[2]:
        message = '   OK : {} slices loaded'.format(volume.shape[2])
    else:
        message = ' *** WARNING -> len(image_list) != slices in ndarray -> check loading'

    return volume, message


def create_stack_light(source_list):
    ''' 
    OLD VERSION - Takes list of source data images.
    Create 'volume', numpy.ndarray(uint8) for load inside it the full stack.
    For every image in list, copy the image inside volume[:,:,image_index] 
    and delete element from list. This reduce memory usage 

    source_list: list of np.ndarray((row, col, channel))
    '''

    # if source_list is not None:
    #     # read images shape and number of slices
    #     i_shape = source_list[0].shape
    #     n_slices = len(source_list)

    #     if n_slices > 0:
    #         # prealloce stack (ndarray, np.uint8)
    #         volume = np.zeros((i_shape[0], i_shape[1], n_slices)).astype(np.uint8)

    #         for z in range(n_slices):
    #             # extract first element and take channell = 0
    #             volume[:,:,z] = (source_list[0][:, :, 0]).astype(np.uint8)
    #             # remove element for free memory
    #             del source_list[0]

    #         if len(source_list) == 0:
    #             return volume
    #         else:
    #             print(' ** WARNING, there is still {} images in source_list'.format(len(source_list)))
    #             return volume
    #     else:
    #         raise ValueError(' Source file list is empty')
    # else:
    #     raise ValueError(' Source file list is None')

    
    ''' 
    NEW VERSION - Takes list of source data images.
    Create 'volume', numpy.ndarray(uint8) for load inside it the full stack.
    For every image in list, copy the image inside a support ndarray and concatenate it at 'volume', 
    then deletes the image from list. This reduce memory usage 

    source_list: list of np.ndarray((row, col, channel))
    '''
    if source_list is not None:
        # read images shape and number of slices
        i_shape = source_list[0].shape
        n_slices = len(source_list)

        if n_slices > 0:
            # create 'volume' (uint8 ndarray) with only one empty slice
            volume = np.zeros((i_shape[0], i_shape[1], 1)).astype(np.uint8)

            # empty ndarray for support
            # (every iteration write one image inside it)
            # then concatenates it with volume
            support = np.zeros((i_shape[0], i_shape[1], 1)).astype(np.uint8)

            # copy first image (only first channel)
            volume[:, :, 0] = source_list[0][:, :, 0].astype(np.uint8)

            # delete first image from list for free memory
            del source_list[0]

            # concatenate all images (only first channel) to 'volume' ndarray
            for z in range(n_slices - 1):
                  
                support[:, :, 0] = source_list[0][:, :, 0].astype(np.uint8)
                del source_list[0]
                volume = np.concatenate((volume, support), axis=2)

            if len(source_list) == 0:
                return volume
            else:
                print(' ** WARNING, there is still {} images in source_list'.format(len(source_list)))
                return volume
        else:
            raise ValueError(' Source file list is empty')
    else:
        raise ValueError(' Source file list is None')


def img_to_array(img_z, channel=0):
    if img_z.mode == 'I;16B':
        X = np.array(img_z)//257
    elif img_z.mode == 'L' or img_z.mode == 'P':
        X = np.array(img_z)
    elif img_z.mode == 'RGB':
        X = np.array(img_z)[:,:,channel]
    else:
        raise ValueError('Dunno how to handle image mode %s' % img_z.mode)
    # return np.flipud(np.rot90(X,1))  # dipende dal sistema di acquisizione
    return X


def stack_to_array(image_file, n_slice=0, channel=0):
    tiff_stack = Image.open(image_file)
    if n_slice > 0:
        n = min(n_slice, tiff_stack.n_frames)
    else:
        n = tiff_stack.n_frames

    # se img_to_array ruota l'immagine, allora serve fare cosi:
    # X = np.zeros(list(tiff_stack.size) + [n])

    # altrimenti: inverto row e col nella shape:
    dim = list(tiff_stack.size)
    X = np.zeros([dim[1], dim[0]] + [n])

    for i in range(n):
        tiff_stack.seek(i)
        X[:,:,i] = img_to_array(tiff_stack, channel)
    return X


def make_dataset(file_list):
   '''
   filename could be either a 3d tiff file or a folder
   '''
   if file_list == None:
       return None
   if len(file_list) == 0:
       return None
   img_orig = []
   for f in natsorted(file_list):
       if os.path.isdir(f):
           for ff in natsorted(os.listdir(f)):
               if ff.endswith('.tif'):
                   img_orig += [stack_to_array(os.path.join(f,ff))]
       else:
           img_orig += [stack_to_array(f)]
   return img_orig


def find_stack_axis(data):
   lengths = np.vstack([d.shape for d in data])
   print(lengths)


def create_stack(d1, d2=None):
    X = None
    if d1 is not None:
        X = np.dstack(d1)
    if d2 is not None:
        X_aux = np.dstack(d2)
        if X is None:
            X = X_aux
        else:
            X = np.dstack([X,X_aux])
    return X
