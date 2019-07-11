import numpy as np
import argparse
import os

from skimage import exposure

from zetastitcher import InputFile
from custom_image_base_tool import save_tiff, normalize, create_img_name_from_index
from custom_tool_kit import nextpow2


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

    # # correct whitespace with backslash
    # given_path = given_path.replace(' ', '\ ')

    # extract base path
    if given_path.endswith('/'):
        given_path = given_path[0:-1]

    return given_path


def CLAHE(parser):
    print('** =========== **')
    print('** START CLAHE **')
    print('** =========== **')
    args = parser.parse_args()

    # extract data
    infile = InputFile(args.source)
    data = infile.whole()

    # sizes
    (num_of_slices, height, weight) = infile.shape
    ksize = nextpow2(weight / 8) if args.kernel == 0 else args.kernel

    # extract path and filename
    base_path = os.path.dirname(os.path.dirname(args.source))
    filename = os.path.splitext(os.path.basename(args.source))[0]

    # create destination paths where save result
    destination_path = os.path.join(base_path, 'clahed_c{}_k{}'.format(args.clip, ksize))
    if not os.path.exists(destination_path):
            os.makedirs(destination_path)

    # print informations
    print('\n ** PATHS : ')
    print(' - source : {}'.format(args.source))
    print(' - output : {}'.format(destination_path))
    print('\n ** FILENAME: {}'.format(filename))
    print('\n ** CLAHE PARAMETERS : ')
    print(' - clip: {}'.format(args.clip))
    print(' - ksize: {}'.format(ksize))

    print('\n ** OUTPUT FORMAT:')
    if args.image_sequence:
        print(' - output is saved like a 2d tiff images sequence')
    else:
        print(' - output is saved ike a 3D tif files')

    # output array
    if not args.image_sequence:
        clahed = np.zeros_like(data)

    print()
    # Execution
    for z in range(num_of_slices):
        img = normalize(data[z, ...])
        img_eq = normalize(exposure.equalize_adapthist(image=img, kernel_size=ksize, clip_limit=args.clip))

        if args.image_sequence:
            img_name = create_img_name_from_index(z, post="_clahe")
            save_tiff(img=img_eq, img_name=img_name, prefix='', comment='', folder_path=destination_path)
            print(img_name)
        else:
            clahed[z, ...] = img_eq
            print('z = {}'.format(z))

    # save output
    if not args.image_sequence:
        save_tiff(clahed, os.path.join(destination_path, 'clahed'))
    print(' \n ** Process Finished \n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLAHE on source path images')
    parser.add_argument('-s', '--source', help='Images to equalize', required=True)
    parser.add_argument('-c', '--clip', help='Clip of CLAHE', required=False, default=0.03, nargs='?', type=float)
    parser.add_argument('-k', '--kernel', help='K, dimension of kernel of CLAHE', required=False, default=0, nargs='?', type=int)
    parser.add_argument('-is', action='store_true', dest='image_sequence', help='if True output is saved like sequence of 2D tiff images, else like a 3d tiff (default, False)')


    CLAHE(parser)
