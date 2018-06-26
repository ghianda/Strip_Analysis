import argparse
import os
from make_data import manage_path_argument

''' 
Main Script of Structural Analysis

parameters: source_folder (of deconvolted images)


Execute:

bla bla bla

'''


def structural_analysis(parser):

    print('\n\n')
    print(' *********************************************************************************')
    print(' ************************* Structural_analysis.py script *************************')
    print(' *********************************************************************************')
    print('\n')

    args = parser.parse_args()
    
    source_path = manage_path_argument(args.source_folder)
    source_path = source_path.replace(' ', '\ ')  # correct whitespace with backslash


    #source_path = args.source_folder
    #
    # if type(source_path) is list:
    #     if len(source_path) > 1:
    #         source_path = ' '.join(source_path)
    #         # source_path = [given_path]  # prepare source_path for 'load_stack_into_numpy_ndarray' function (it takes a list in input)
    #     else:
    #         source_path = source_path[0]

    # # correct whitespace with backslash
    # source_path = source_path.replace(' ', '\ ')

    # # extract base path
    # if source_path.endswith('/'):
    #     source_path = source_path[0:-1]

    # extract base path and stack name
    base_path = os.path.dirname(os.path.dirname(source_path))
    stack_name = os.path.basename(source_path)

    # create folder path of binary mask images
    mask_path = os.path.join(base_path, 'mask_bin', stack_name)
    # mask_path = mask_path.replace(' ', '\ ')  # correct white space ' '  with '\ '

    # create folder path of segmented images
    segmented_path = os.path.join(base_path, 'segmented', stack_name)
    # segmented_path = segmented_path.replace(' ', '\ ')  # correct white space ' '  with '\ '

    # call  ALFA_volume_analysis script
    print(' ---------> call ALFA')
    os.system('python3 ALFA_volume_analysis.py -sf {}'.format(source_path))

    # call  BETA_estimated_section script
    # todo renderlo opzionale con input d'ingresso
    print(' ---------> call BETA')
    os.system('python3 BETA_estimated_section.py -sf {}'.format(mask_path))

    # call  ALFA_volume_analysis script
    print(' ---------> call GAMMA')
    os.system('python3 GAMMA_orientation_analysis.py -sf {}'.format(segmented_path))


def main():
    parser = argparse.ArgumentParser(
        description='Strip images enhanced and 3D Structural analysis',
        epilog='Author: Francesco Giardini <giardini@lens.unifi.it>',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-sf', '--source_folder', nargs='+', required=False, help='input images path')
    structural_analysis(parser)

if __name__ == '__main__':
    main()


